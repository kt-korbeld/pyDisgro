"""
Analytic tripeptide loop closure.

Port of ``loop_closure_c/tripep_closure.h`` -- the method of Coutsias, Seok,
Jacobson and Dill: given the anchor atoms N1/CA1 and CA3/C3 of a tripeptide and
ideal bond geometry, the closed conformations are the real roots of a degree-16
polynomial, of which there are at most 16.

The reference implementation solves that polynomial with a hand-written Sturm
sequence and bisection (``sturm.h``).  This port uses ``numpy.roots`` instead:
the root sets agree to numerical tolerance, the code is ~600 lines shorter, and
the companion-matrix eigenvalue solve is at least as robust.  Individual
conformations will therefore differ from a C++ run in the last digits.
"""

import math
import numpy as np

from .constants import *
from .energy import one_res_en
from .geom import SampleOne, calCo_batch, rng, torsion_batch
from .potential import PF
from .residue import Residue

# ---------------------------------------------------------------------------
# helper functions for fast math with polynomials
# ---------------------------------------------------------------------------

def _pmul1(u1, u2):
    """
    Product of two 1-D polynomials, truncated back to 17 coefficients.
    """
    return np.convolve(u1, u2)[:17]


def _pmul_sub1(u1, u2, u3, u4):
    return _pmul1(u1, u2) - _pmul1(u3, u4)


def _pmul2(u1, u2):
    """
    Product of two bivariate polynomials held as 5x5 coefficient arrays.
    Packing each array into a 9x9 grid and flattening turns the 2-D
    convolution into a single 1-D one: with both degrees at most 4 per
    variable, no term can carry from one row into the next.
    """
    a = np.zeros((9, 9))
    b = np.zeros((9, 9))
    a[:5, :5] = u1
    b[:5, :5] = u2
    c = np.convolve(a.ravel(), b.ravel())[:81].reshape(9, 9)
    return c[:5, :5]


def _pmul_sub2(u1, u2, u3, u4):
    return _pmul2(u1, u2) - _pmul2(u3, u4)


# the analytic tripeptide closure requires many repeated calculations on single coordinates.
# for these short vectors, pre-defined operations on python tuples are faster than numpy arrays

def _dot(a, b):
    """
    return dot product of two 3d coordinates a and b
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def _cross(a, b):
    """
    return cross product of two 3d coordinates a and b
    """
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])

def _sub(a, b):
    """
    subtract 3d coordinates a and b
    """
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

def _neg(a):
    """
    return negative of 3d coordinate a
    """
    return (-a[0], -a[1], -a[2])

def _scale(a, s):
    """
    scale 3d coordinate a by factor s
    """
    return (a[0] * s, a[1] * s, a[2] * s)

def _calc_bnd_ang(r1, r2):
    """
    Angle between two unit vectors.
    """
    arg = _dot(r1, r2)
    return math.acos(math.copysign(min(abs(arg), 1.0), arg))

def _calc_dih_ang(r1, r2, r3):
    """
    Dihedral defined by three consecutive unit bond vectors.
    """
    p = _cross(r1, r2)
    q = _cross(r2, r3)
    s = _cross(r3, r1)
    arg = _dot(p, q) / math.sqrt(_dot(p, p) * _dot(q, q))
    arg = math.copysign(min(abs(arg), 1.0), arg)
    return math.copysign(math.acos(arg), _dot(s, r2))

def _quaternion(axis, quarter_ang):
    tan_w = math.tan(quarter_ang)
    tan_sqr = tan_w * tan_w
    tan1 = 1.0 + tan_sqr
    cosine = (1.0 - tan_sqr) / tan1
    sine = 2.0 * tan_w / tan1
    return (cosine, axis[0] * sine, axis[1] * sine, axis[2] * sine)


def _rotation_matrix(q):
    q0, q1, q2, q3 = q
    b0, b1, b2, b3 = 2.0 * q0, 2.0 * q1, 2.0 * q2, 2.0 * q3
    q00 = b0 * q0 - 1.0
    q01, q02, q03 = b0 * q1, b0 * q2, b0 * q3
    q11, q12, q13 = b1 * q1, b1 * q2, b1 * q3
    q22, q23, q33 = b2 * q2, b2 * q3, b3 * q3
    return ((q00 + q11, q12 - q03, q13 + q02),
            (q12 + q03, q00 + q22, q23 - q01),
            (q13 - q02, q23 + q01, q00 + q33))


def _matmul(U, v):
    return (U[0][0] * v[0] + U[0][1] * v[1] + U[0][2] * v[2],
            U[1][0] * v[0] + U[1][1] * v[1] + U[1][2] * v[2],
            U[2][0] * v[0] + U[2][1] * v[1] + U[2][2] * v[2])


# ---------------------------------------------------------------------------
# the closure solver
# ---------------------------------------------------------------------------

class TripeptideClosure:
    """
    Solver for one set of bond lengths, angles and peptide torsions.
    initialize() corresponds to initialize_loop_closure() and only depends
    on the ideal geometry. solve() corresponds to solve_3pep_poly() and is
    called once per closure attempt with the four anchor atoms.
    """

    def __init__(self):
        self.len0 = np.zeros(6)
        self.b_ang0 = np.zeros(7)
        self.t_ang0 = np.zeros(2)
        self.aa13_min_sqr = 0.0
        self.aa13_max_sqr = 0.0
        self.delta = np.zeros(4)
        self.xi = np.zeros(3)
        self.eta = np.zeros(3)
        self.alpha = np.zeros(3)
        self.theta = np.zeros(3)
        self.len_aa = np.zeros(3)
        self.len_na = np.zeros(3)
        self.len_ac = np.zeros(3)
        self.C0 = np.zeros((3, 3))
        self.C1 = np.zeros((3, 3))
        self.C2 = np.zeros((3, 3))
        self.Q = np.zeros((5, 5))
        self.R = np.zeros((3, 17))

    # setup
    def initialize(self, b_len, b_ang, t_ang):
        """
        Precompute the fixed-geometry quantities (``tripep_closure.h:175``).
        """
        self.len0 = tuple(float(v) for v in b_len)
        self.b_ang0 = tuple(float(v) for v in b_ang)
        self.t_ang0 = tuple(float(v) for v in t_ang)

        axis = (1.0, 0.0, 0.0)
        len0, b_ang0, t_ang0 = self.len0, self.b_ang0, self.t_ang0

        for i in range(2):
            rr_a1 = (math.cos(b_ang0[3 * i + 1]) * len0[3 * i],
                     math.sin(b_ang0[3 * i + 1]) * len0[3 * i], 0.0)
            rr_n2 = (len0[3 * i + 1], 0.0, 0.0)
            rr_c1a1 = rr_a1                       # rr_c1 is the origin
            rr_n2a2_ref = (-math.cos(b_ang0[3 * i + 2]) * len0[3 * i + 2],
                           math.sin(b_ang0[3 * i + 2]) * len0[3 * i + 2], 0.0)
            Us = _rotation_matrix(_quaternion(axis, t_ang0[i] * 0.25))
            m = _matmul(Us, rr_n2a2_ref)
            rr_a2 = (m[0] + rr_n2[0], m[1] + rr_n2[1], m[2] + rr_n2[2])
            rr_a1a2 = _sub(rr_a2, rr_a1)
            len1 = math.sqrt(_dot(rr_a1a2, rr_a1a2))
            self.len_aa[i + 1] = len1

            bb_c1a1 = _scale(rr_c1a1, 1.0 / len0[3 * i])
            bb_a1a2 = _scale(rr_a1a2, 1.0 / len1)
            bb_a2n2 = _scale(_sub(rr_n2, rr_a2), 1.0 / len0[3 * i + 2])

            self.xi[i + 1] = _calc_bnd_ang(_neg(bb_a1a2), bb_a2n2)
            self.eta[i] = _calc_bnd_ang(bb_a1a2, _neg(bb_c1a1))
            self.delta[i + 1] = PI - _calc_dih_ang(bb_c1a1, bb_a1a2, bb_a2n2)

        a_min = self.b_ang0[3] - (self.xi[1] + self.eta[1])
        a_max = min(self.b_ang0[3] + (self.xi[1] + self.eta[1]), PI)
        l1, l2 = self.len_aa[1], self.len_aa[2]
        self.aa13_min_sqr = l1 ** 2 + l2 ** 2 - 2.0 * l1 * l2 * math.cos(a_min)
        self.aa13_max_sqr = l1 ** 2 + l2 ** 2 - 2.0 * l1 * l2 * math.cos(a_max)

    # execute closure
    def _get_input_angles(self, r_n1, r_a1, r_a3, r_c3):
        """
        Set up the cone geometry for these anchors. False if unreachable.
        """
        r_a1a3 = _sub(r_a3, r_a1)
        dr_sqr = _dot(r_a1a3, r_a1a3)
        # Cheapest possible rejection, and by far the most common outcome
        # during the jittered retry loop: the anchors are simply too far apart
        # (or too close) for any tripeptide with this geometry to span them.
        if dr_sqr < self.aa13_min_sqr or dr_sqr > self.aa13_max_sqr:
            return False
        self.len_aa[0] = math.sqrt(dr_sqr)

        r_a1n1 = _sub(r_n1, r_a1)
        self.len_na[0] = math.sqrt(_dot(r_a1n1, r_a1n1))
        self.len_na[1] = self.len0[2]
        self.len_na[2] = self.len0[5]
        r_a3c3 = _sub(r_c3, r_a3)
        self.len_ac[0] = self.len0[0]
        self.len_ac[1] = self.len0[3]
        self.len_ac[2] = math.sqrt(_dot(r_a3c3, r_a3c3))

        self.r_a1n1 = r_a1n1
        b_a1n1 = _scale(r_a1n1, 1.0 / self.len_na[0])
        b_a3c3 = _scale(r_a3c3, 1.0 / self.len_ac[2])
        b_a1a3 = _scale(r_a1a3, 1.0 / self.len_aa[0])
        self.b_a1n1, self.b_a3c3, self.b_a1a3 = b_a1n1, b_a3c3, b_a1a3

        self.delta[3] = _calc_dih_ang(_neg(b_a1n1), b_a1a3, b_a3c3)
        self.delta[0] = self.delta[3]
        self.xi[0] = _calc_bnd_ang(_neg(b_a1a3), b_a1n1)
        self.eta[2] = _calc_bnd_ang(b_a1a3, b_a3c3)

        self.cos_delta = np.cos(self.delta)
        self.sin_delta = np.sin(self.delta)
        self.cos_delta[0] = self.cos_delta[3]
        self.sin_delta[0] = self.sin_delta[3]
        self.cos_xi = np.cos(self.xi)
        self.sin_xi = np.sin(self.xi)
        self.cos_eta = np.cos(self.eta)
        self.sin_eta = np.sin(self.eta)

        self.theta = np.array([self.b_ang0[0], self.b_ang0[3], self.b_ang0[6]])
        self.cos_theta = np.cos(self.theta)

        laa = self.len_aa
        self.cos_alpha = np.zeros(3)
        self.sin_alpha = np.zeros(3)
        self.cos_alpha[0] = -(laa[0] ** 2 + laa[1] ** 2 - laa[2] ** 2) / (2.0 * laa[0] * laa[1])
        self.alpha[0] = math.acos(float(np.clip(self.cos_alpha[0], -1.0, 1.0)))
        self.sin_alpha[0] = math.sin(self.alpha[0])
        self.cos_alpha[1] = (laa[1] ** 2 + laa[2] ** 2 - laa[0] ** 2) / (2.0 * laa[1] * laa[2])
        self.alpha[1] = math.acos(float(np.clip(self.cos_alpha[1], -1.0, 1.0)))
        self.sin_alpha[1] = math.sin(self.alpha[1])
        self.alpha[2] = PI - self.alpha[0] + self.alpha[1]
        self.cos_alpha[2] = math.cos(self.alpha[2])
        self.sin_alpha[2] = math.sin(self.alpha[2])

        # Two-cone existence test: a solution needs |alpha - theta| <= xi + eta.
        for i in range(3):
            if abs(self.alpha[i] - self.theta[i]) > (self.xi[i] + self.eta[i]):
                return False
        return True

    def _get_poly_coeff(self):
        """
        Build the degree-16 polynomial (``tripep_closure.h:601``).
        """
        B = np.zeros((9, 3))
        for i in range(3):
            A0 = self.cos_alpha[i] * self.cos_xi[i] * self.cos_eta[i] - self.cos_theta[i]
            A1 = -self.sin_alpha[i] * self.cos_xi[i] * self.sin_eta[i]
            A2 = self.sin_alpha[i] * self.sin_xi[i] * self.cos_eta[i]
            A3 = self.sin_xi[i] * self.sin_eta[i]
            A4 = A3 * self.cos_alpha[i]
            cd, sd = self.cos_delta[i], self.sin_delta[i]
            A21, A22 = A2 * cd, A2 * sd
            A31, A32 = A3 * cd, A3 * sd
            A41, A42 = A4 * cd, A4 * sd
            B[0, i] = A0 + A22 + A31
            B[1, i] = 2.0 * (A1 + A42)
            B[2, i] = 2.0 * (A32 - A21)
            B[3, i] = -4.0 * A41
            B[4, i] = A0 + A22 - A31
            B[5, i] = A0 - A22 - A31
            B[6, i] = -2.0 * (A21 + A32)
            B[7, i] = 2.0 * (A1 - A42)
            B[8, i] = A0 - A22 + A31

        C0, C1, C2 = self.C0, self.C1, self.C2
        C0[0] = [B[0, 0], B[2, 0], B[5, 0]]
        C1[0] = [B[1, 0], B[3, 0], B[7, 0]]
        C2[0] = [B[4, 0], B[6, 0], B[8, 0]]
        for i in (1, 2):
            C0[i] = [B[0, i], B[1, i], B[4, i]]
            C1[i] = [B[2, i], B[3, i], B[6, i]]
            C2[i] = [B[5, i], B[7, i], B[8, i]]

        u11 = np.zeros((5, 5)); u11[0, :3] = C0[0]
        u12 = np.zeros((5, 5)); u12[0, :3] = C1[0]
        u13 = np.zeros((5, 5)); u13[0, :3] = C2[0]
        u31 = np.zeros((5, 5)); u31[:3, 0] = C0[1]
        u32 = np.zeros((5, 5)); u32[:3, 0] = C1[1]
        u33 = np.zeros((5, 5)); u33[:3, 0] = C2[1]

        um1 = _pmul_sub2(u32, u32, u31, u33)
        um2 = _pmul_sub2(u12, u32, u11, u33)
        um3 = _pmul_sub2(u12, u33, u13, u32)
        um4 = _pmul_sub2(u11, u33, u31, u13)
        um5 = _pmul_sub2(u13, um1, u33, um2)
        um6 = _pmul_sub2(u13, um4, u12, um3)
        self.Q = _pmul_sub2(u11, um5, u31, um6)

        R = np.zeros((3, 17))
        R[0, :3] = C0[2]
        R[1, :3] = C1[2]
        R[2, :3] = C2[2]
        self.R = R

        Qp = np.zeros((5, 17))
        Qp[:, :5] = self.Q

        f1 = _pmul_sub1(R[1], R[1], R[0], R[2])
        f2 = _pmul1(R[1], R[2])
        f3 = _pmul_sub1(R[1], f1, R[0], f2)
        f4 = _pmul1(R[2], f1)
        f5 = _pmul_sub1(R[1], f3, R[0], f4)

        f6 = _pmul_sub1(Qp[1], R[1], Qp[0], R[2])
        f7 = _pmul_sub1(Qp[2], f1, R[2], f6)
        f8 = _pmul_sub1(Qp[3], f3, R[2], f7)
        f9 = _pmul_sub1(Qp[4], f5, R[2], f8)

        f10 = _pmul_sub1(Qp[3], R[1], Qp[4], R[0])
        f11 = _pmul_sub1(Qp[2], f1, R[0], f10)
        f12 = _pmul_sub1(Qp[1], f3, R[0], f11)

        f13 = _pmul_sub1(Qp[2], R[1], Qp[1], R[2])
        f14 = _pmul_sub1(Qp[3], f1, R[2], f13)
        f15 = _pmul_sub1(Qp[3], R[1], Qp[2], R[2])
        f16 = _pmul_sub1(Qp[4], f1, R[2], f15)
        f17 = _pmul_sub1(Qp[1], f14, Qp[0], f16)

        f18 = _pmul_sub1(Qp[2], R[2], Qp[3], R[1])
        f19 = _pmul_sub1(Qp[1], R[2], Qp[3], R[0])
        f20 = _pmul_sub1(Qp[3], f19, Qp[2], f18)
        f21 = _pmul_sub1(Qp[1], R[1], Qp[2], R[0])
        f22 = _pmul1(Qp[4], f21)
        f23 = f20 - f22
        f24 = _pmul1(R[0], f23)
        f25 = f17 - f24
        f26 = _pmul_sub1(Qp[4], f12, R[2], f25)
        poly = _pmul_sub1(Qp[0], f9, R[0], f26)

        if poly[16] < 0.0:
            poly = -poly
        return poly

    def _calc_t2(self, t0):
        powers = np.array([1.0, t0, t0 ** 2, t0 ** 3, t0 ** 4])
        A = self.Q @ powers
        B = self.R[:, :3] @ powers[:3]
        B0, B1, B2 = B
        B2_2 = B2 * B2
        K0 = A[2] * B2 - A[4] * B0
        K1 = A[3] * B2 - A[4] * B1
        K2 = A[1] * B2_2 - K1 * B0
        K3 = K0 * B2 - K1 * B1
        return (K3 * B0 - A[0] * B2_2 * B2) / (K2 * B2 - K3 * B1)

    def _calc_t1(self, t0, t2):
        p0 = np.array([1.0, t0, t0 ** 2])
        p2 = np.array([1.0, t2, t2 ** 2])
        U11 = float(self.C0[0] @ p0)
        U12 = float(self.C1[0] @ p0)
        U13 = float(self.C2[0] @ p0)
        U31 = float(self.C0[1] @ p2)
        U32 = float(self.C1[1] @ p2)
        U33 = float(self.C2[1] @ p2)
        return (U31 * U13 - U11 * U33) / (U12 * U33 - U13 * U32)

    def _coord_from_roots(self, roots, r_n1, r_a1, r_a3, r_c3):
        """
        Turn polynomial roots into backbone coordinates (``:1115``).
        """
        ex = np.asarray(self.b_a1a3)
        ez = np.asarray(_cross(self.r_a1n1, self.b_a1a3))
        ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)

        ca, sa = self.cos_alpha, self.sin_alpha
        b_a1a2 = -ca[0] * ex + sa[0] * ey
        b_a3a2 = ca[2] * ex + sa[2] * ey

        p_s = np.array([-ex, -b_a1a2, b_a3a2])
        s1 = np.array([ez, -ez, ez])
        t2_0 = sa[0] * ex + ca[0] * ey
        t2_1 = sa[2] * ex - ca[2] * ey
        s2 = np.array([ey, t2_0, t2_1])
        p_t = np.array([b_a1a2, -b_a3a2, ex])
        t1 = np.array([ez, -ez, ez])
        t2 = np.array([t2_0, t2_1, -ey])

        p_s_c = p_s * self.cos_xi[:, None]
        s1_s = s1 * self.sin_xi[:, None]
        s2_s = s2 * self.sin_xi[:, None]
        p_t_c = p_t * self.cos_eta[:, None]
        t1_s = t1 * self.sin_eta[:, None]
        t2_s = t2 * self.sin_eta[:, None]

        r_tmp = (np.asarray(self.r_a1n1) / self.len_na[0] - p_s_c[0]) / self.sin_xi[0]
        sig1_init = math.copysign(_calc_bnd_ang(s1[0], r_tmp),
                                  float(np.dot(r_tmp, s2[0])))

        r_a = np.array([r_a1, r_a1 + self.len_aa[1] * b_a1a2, r_a3])
        r0 = r_a1

        out_n = np.zeros((len(roots), 3, 3))
        out_a = np.zeros((len(roots), 3, 3))
        out_c = np.zeros((len(roots), 3, 3))

        for k, root in enumerate(roots):
            t0 = float(root)
            t2v = self._calc_t2(t0)
            t1v = self._calc_t1(t0, t2v)
            half_tan = np.array([t1v, t2v, t0])

            cos_tau = np.zeros(4)
            sin_tau = np.zeros(4)
            for i in range(1, 4):
                ht = half_tan[i - 1]
                tmp = 1.0 + ht * ht
                cos_tau[i] = (1.0 - ht * ht) / tmp
                sin_tau[i] = 2.0 * ht / tmp
            cos_tau[0] = cos_tau[3]
            sin_tau[0] = sin_tau[3]

            cos_sig = self.cos_delta[:3] * cos_tau[:3] + self.sin_delta[:3] * sin_tau[:3]
            sin_sig = self.sin_delta[:3] * cos_tau[:3] - self.cos_delta[:3] * sin_tau[:3]

            r_s = p_s_c + cos_sig[:, None] * s1_s + sin_sig[:, None] * s2_s
            r_t = p_t_c + cos_tau[1:4, None] * t1_s + sin_tau[1:4, None] * t2_s
            r_n = r_s * self.len_na[:, None] + r_a
            r_c = r_t * self.len_ac[:, None] + r_a

            sig1 = math.atan2(sin_sig[0], cos_sig[0])
            Us = _rotation_matrix(_quaternion(-ex, -(sig1 - sig1_init) * 0.25))

            out_n[k, 0] = r_n1
            out_a[k, 0] = r_a1
            out_c[k, 0] = Us @ (r_c[0] - r0) + r0
            out_n[k, 1] = Us @ (r_n[1] - r0) + r0
            out_a[k, 1] = Us @ (r_a[1] - r0) + r0
            out_c[k, 1] = Us @ (r_c[1] - r0) + r0
            out_n[k, 2] = Us @ (r_n[2] - r0) + r0
            out_a[k, 2] = r_a3
            out_c[k, 2] = r_c3
        return out_n, out_a, out_c

    def solve(self, r_n1, r_a1, r_a3, r_c3, tol=1e-8):
        """
        Return (r_n, r_a, r_c) arrays of shape (n_soln, 3, 3).
        """
        empty = (np.zeros((0, 3, 3)),) * 3
        if not self._get_input_angles(tuple(map(float, r_n1)),
                                      tuple(map(float, r_a1)),
                                      tuple(map(float, r_a3)),
                                      tuple(map(float, r_c3))):
            return empty

        poly = self._get_poly_coeff()
        if not np.isfinite(poly).all() or poly[16] == 0.0:
            return empty

        # numpy.roots wants the highest-degree coefficient first.
        with np.errstate(all="ignore"):
            roots = np.roots(poly[::-1])
        if roots.size == 0:
            return empty
        scale = np.maximum(1.0, np.abs(roots))
        real = roots[np.abs(roots.imag) < tol * scale].real
        if real.size == 0:
            return empty
        # Sorted ascending, which is the order the C++ Sturm bisection returns
        # them in; keeps solution indices comparable between the two codes.
        real = np.sort(real)[:MAX_SOLN]

        with np.errstate(all="ignore"):
            out = self._coord_from_roots(real, np.asarray(r_n1, float),
                                         np.asarray(r_a1, float),
                                         np.asarray(r_a3, float),
                                         np.asarray(r_c3, float))
        good = np.isfinite(out[0]).all(axis=(1, 2))
        good &= np.isfinite(out[1]).all(axis=(1, 2))
        good &= np.isfinite(out[2]).all(axis=(1, 2))
        return out[0][good], out[1][good], out[2][good]


# ---------------------------------------------------------------------------
# structure-level driver
# ---------------------------------------------------------------------------

_solver = TripeptideClosure()

# Ideal bond geometry per residue-type triple; the tables never change, and
# the jittered closure asks for the same triple hundreds of times per trial.
_IDEAL_GEOMETRY = {}


def analytic_closure(flatstruc, start, Start, End, jitter=None):
    """
    Close the tripeptide at residues ``start to start+2`` 
    in the input flatstruc object.
    Port of ``Structure::analyticClosure`` (structure.cpp:1353).  With
    ``jitter=(len_change, bond_change, torsion_change)`` the ideal bond
    lengths, bond angles and peptide torsions are drawn uniformly within those
    half-widths, reproducing ``Structure::analyticClosure_h`` -- the C++ calls
    that variant up to 300 times when the plain closure fails.

    Returns True if a solution was found and written into the input flatstruc.
    """
    rt = flatstruc.res_type
    key = (int(rt[start]), int(rt[start + 1]), int(rt[start + 2]))
    ideal = _IDEAL_GEOMETRY.get(key)
    if ideal is None:
        t0, t1, t2 = key
        bl, ba = Residue.bond_length, Residue.bond_angle
        ideal = (np.array([bl[t0][ATM_C], bl[t1][ATM_N], bl[t1][ATM_CA],
                           bl[t1][ATM_C], bl[t2][ATM_N], bl[t2][ATM_CA]]),
                 np.array([ba[t0][ATM_C], ba[t1][ATM_N], ba[t1][ATM_CA],
                           ba[t1][ATM_C], ba[t2][ATM_N], ba[t2][ATM_CA],
                           ba[t2][ATM_C]]))
        _IDEAL_GEOMETRY[key] = ideal
    b_len, b_ang = ideal
    t_ang = np.array([PI, PI])

    if jitter is not None:
        dl, db, dt = jitter
        # One draw for all fifteen parameters: this runs hundreds of times per
        # trial, so the per-call RNG overhead matters more than the arithmetic.
        u = rng().random(15)
        b_len = b_len + (2.0 * u[:6] - 1.0) * dl
        b_ang = b_ang + (2.0 * u[6:13] - 1.0) * db
        t_ang = t_ang + (2.0 * u[13:] - 1.0) * dt
        t_ang = np.where(t_ang > PI, t_ang - 2 * PI, t_ang)

    _solver.initialize(b_len, b_ang, t_ang)
    r_n, r_a, r_c = _solver.solve(flatstruc.atom(start, ATM_N),
                                  flatstruc.atom(start, ATM_CA),
                                  flatstruc.atom(start + 2, ATM_CA),
                                  flatstruc.atom(start + 2, ATM_C))
    n_soln = r_n.shape[0]
    if n_soln == 0:
        return False

    # Score every solution, keeping the coordinates each one implies.
    energies = np.zeros(n_soln)
    saved = np.zeros((n_soln, 3, NUM_BB_ATOM, 3))
    # Glycine has no CB slot, so only copy the slots the residue actually has.
    n_slot = [min(NUM_BB_ATOM, int(flatstruc.res_natom[start + i])) for i in range(3)]
    for k in range(n_soln):
        _place_solution(flatstruc, start, r_n[k], r_a[k], r_c[k])
        for i in range(3):
            lo = int(flatstruc.res_start[start + i])
            saved[k, i, :n_slot[i]] = flatstruc.xyz[lo:lo + n_slot[i]]
        energies[k] = _score_solution(flatstruc, start, Start, End)

    minE = energies.min()
    prob = np.power(EXPO, (minE - energies) * 0.5)
    if not np.isfinite(prob).all():
        raise FloatingPointError(f"closure probability overflow: {energies}")
    chosen = (int(rng().integers(0, n_soln)) if prob.sum() == 0
              else SampleOne(prob))

    for i in range(3):
        lo = int(flatstruc.res_start[start + i])
        flatstruc.xyz[lo:lo + n_slot[i]] = saved[chosen, i, :n_slot[i]]
        flatstruc.update_center(start + i, sidechain=False)
    return True


def _place_solution(flatstruc, start, r_n, r_a, r_c):
    """
    Write one closure solution plus the derived O and CB atoms.
    """
    for i in range(3):
        res = start + i
        flatstruc.xyz[flatstruc.index(res, ATM_N)] = r_n[i]
        flatstruc.xyz[flatstruc.index(res, ATM_CA)] = r_a[i]
        flatstruc.xyz[flatstruc.index(res, ATM_C)] = r_c[i]

    for i in range(3):
        res = start + i
        n = flatstruc.atom(res, ATM_N)
        ca = flatstruc.atom(res, ATM_CA)
        c = flatstruc.atom(res, ATM_C)
        nxt = flatstruc.atom(res + 1, ATM_N)
        psi = math.radians(float(torsion_batch(n, ca, c, nxt)))
        rtype = flatstruc.res_type[res]
        flatstruc.xyz[flatstruc.index(res, ATM_O)] = calCo_batch(
            n, ca, c, Residue.bond_length[rtype][ATM_O],
            Residue.bond_angle[rtype][ATM_O], psi + PI)
        if rtype != GLY:
            flatstruc.xyz[flatstruc.index(res, ATM_CB)] = calCo_batch(
                n, c, ca, Residue.bond_length[rtype][ATM_CB],
                Residue.bond_angle[rtype][ATM_CB], PI * 122.55 / 180)
        flatstruc.update_center(res, sidechain=False)


def _score_solution(flatstruc, start, Start, End):
    """
    LOODIS energy of the three closed residues (structure.cpp:1521-1552).
    """
    total = 0.0
    end_lo = int(flatstruc.res_start[End])
    end_pts = flatstruc.xyz[end_lo + ATM_CA:end_lo + ATM_C + 1]        # CA and C
    end_types = flatstruc.atype[end_lo + ATM_CA:end_lo + ATM_C + 1]

    for i in range(3):
        res = start + i
        lo = int(flatstruc.res_start[res])
        n_slot = min(NUM_BB_ATOM, int(flatstruc.res_natom[res]))
        cand = np.zeros((1, NUM_BB_ATOM, 3))
        ctypes = np.full(NUM_BB_ATOM, UNDEF, dtype=np.int64)
        cand[0, :n_slot] = flatstruc.xyz[lo:lo + n_slot]
        ctypes[:n_slot] = flatstruc.atype[lo:lo + n_slot]
        ref = flatstruc.res_bbc[res][None, :]

        total += one_res_en(flatstruc, cand, ctypes, ref, res, 1, res - 2,
                            Start, End, 1)[0]
        if End != flatstruc.numRes:
            total += one_res_en(flatstruc, cand, ctypes, ref, res, End + 1,
                                flatstruc.numRes, Start, End, 2)[0]

        # Explicit term against the CA and C of the anchor residue End.  The
        # C++ runs this over every slot of the residue and applies no
        # coordinate check, so it is reproduced verbatim.
        nat = int(flatstruc.res_natom[res])
        atoms = flatstruc.xyz[lo:lo + nat]
        types = flatstruc.atype[lo:lo + nat]
        keep = (types != UNDEF) & (types < H_ATOM_TYPE)
        if not keep.any():
            continue
        d2 = np.sum((atoms[keep][:, None, :] - end_pts[None, :, :]) ** 2, axis=-1)
        close = d2 <= PF_DIS_CUT_SQUARE
        if not close.any():
            continue
        bins = np.clip((np.sqrt(d2) / H_INLO).astype(np.int64), 0,
                       LOODIS_DIS_BIN - 1)
        ta = np.broadcast_to(types[keep][:, None], d2.shape)
        tb = np.broadcast_to(end_types[None, :], d2.shape)
        total += float(PF.LOODIS[ta[close] - 1, tb[close] - 1, bins[close]].sum())
    return total
