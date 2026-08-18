# samples empirical distributions of backbone configurations

import os
import numpy as np

from .constants import *
from .geom import box_muller_batch, frand, rng
from .residue import Rotamer, SCR

class EmpiricalDistances:
    """
    The conditional end-to-end distance distributions from the frag.*.txt files
    label 0 is the N-atom table frag.N.C, 1 the C-atom table frag.C.CA.  
    For each fragment length the file gives a four-number header:
    min dist, dist bin width, min dist decrease, decrease bin width
    followed by a 32x32 bin density.
    """

    def __init__(self):
        self.eted = np.zeros((2, MAX_FRAG_LEN, N_DIST_BIN, N_DIST_BIN))
        self.min_dis = np.zeros((2, MAX_FRAG_LEN))
        self.dis_by = np.zeros((2, MAX_FRAG_LEN))
        self.min_del = np.zeros((2, MAX_FRAG_LEN))
        self.del_by = np.zeros((2, MAX_FRAG_LEN))

    def load(self, filename, label):
        """
        Port of SMC::fragdis
        """
        frag = -1
        row = 0
        with open(filename) as f:
            for line in f:
                if not line or line[0] == "#":
                    continue
                linedata = line.split()
                if len(linedata) == 4:
                    frag += 1
                    row = 0
                    (self.min_dis[label][frag], self.dis_by[label][frag],
                     self.min_del[label][frag], self.del_by[label][frag]) = map(float, linedata)
                elif len(linedata) == N_DIST_BIN:
                    self.eted[label][frag][row] = [float(v) for v in linedata]
                    row += 1

    @classmethod
    def load_default(cls):
        """
        Load in the default empirical distance files
        """
        obj = cls()
        obj.load(data_path(FILE_FRAG_N_C), 0)
        obj.load(data_path(FILE_FRAG_C_CA), 1)
        return obj

    def range(self, label, length):
        """
        Return the valid min and max distance range of the loaded data.
        """
        lo = self.min_dis[label][length]
        return lo, lo + 31 * self.dis_by[label][length]


def load_joint_angles(filename=None):
    """
    Joint phi/psi counts per residue type. Port of SMC::simpBBT_Init.
    Returns an (20, TORBIN, TORBIN) array of observed counts, 
    used to bias the distance-sampled proposals towards allowed Ramachandran regions.
    """
    if filename is None:
        filename = data_path(FILE_BBT)
    if not os.path.exists(filename):
        raise FileNotFoundError(f"cannot open BBT sampling file {filename}")
    joint = np.zeros((20, TORBIN, TORBIN), dtype=np.int64)
    with open(filename) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            linedata = line.split("\t")
            if len(linedata) != 4:
                continue
            res = int(linedata[0])
            phi = int(linedata[1]) + 180
            psi = int(linedata[2]) + 180
            joint[res][phi // BBTbinSize][psi // BBTbinSize] = int(linedata[3])
    return joint


# NOTE: the C++ also loads data/LoopGeo_37_pdf_21.txt via SMC::geometryinfo and
# defines SMC::geometryProb to use it, but geometryProb is never called from
# anywhere in the program.  Both are omitted here.


# ---------------------------------------------------------------------------
# distance-guided atom placement
# ---------------------------------------------------------------------------

def sample_distance(dist, b, c, B, theta, lcd, lcon, label, rem, n_states,
                    threshold=None):
    """
    Place an atom by sampling its distance to the loop anchor.
    Port of SMC::sample_distanced. empirical distances are sampled from 
    dist, which should be the EmpiricalDistances class with loaded data.
    The new atom must sit on the circle of points at bond length
    lcd from c making bond angle theta with b-c.  
    Its distance to the anchor B must be [|mm-B|, |mp-B|] DiSGro draws
    the decrease in the end-to-end distance from the empirical table, then
    solves for the position on the circle.
    theta: bond angle in radians, lcd: bond length of new atoms
    lcon: current distance from current atom to anchor
    label: Which empirical table to use, n_states: number of states to return
    Returns positions, rows are NaN where the geometry had no solution.
    """
    b = np.broadcast_to(np.asarray(b, float), (n_states, 3)) # preceding atom
    c = np.broadcast_to(np.asarray(c, float), (n_states, 3)) # current atom
    B = np.broadcast_to(np.asarray(B, float), (n_states, 3)) #anchor
    lcon = np.broadcast_to(np.asarray(lcon, float), (n_states,)) # dist from c to B

    cb = c - b
    cb_sq = np.einsum("ij,ij->i", cb, cb)
    lbc = np.sqrt(cb_sq)
    # e: centre of the circle of allowed positions
    e = c - cb * (lcd * np.cos(theta) / lbc)[:, None]
    Be = B - e
    #Bp: B projected onto the plane of the circle.
    Bp = B - cb * (np.einsum("ij,ij->i", Be, cb) / cb_sq)[:, None]
    Bpe = Bp - e
    len_Bpe = np.sqrt(np.einsum("ij,ij->i", Bpe, Bpe))
    step = (lcd * np.sin(theta) / len_Bpe)[:, None] * Bpe
    mm = e + step
    mp = e - step

    d_mp = mp - B
    d_mm = mm - B
    md0 = lcon - np.sqrt(np.einsum("ij,ij->i", d_mp, d_mp))
    md1 = lcon - np.sqrt(np.einsum("ij,ij->i", d_mm, d_mm))

    y = _sample_decrement(dist, label, rem, lcon, md0, md1)
    d = lcon - y

    # Solve for the angle on the circle that gives distance d to the anchor.
    lcd_sin = lcd * np.sin(theta)
    with np.errstate(invalid="ignore", divide="ignore"):
        arg = ((np.einsum("ij,ij->i", Be, Be) + lcd_sin ** 2 - d ** 2)
               / (2.0 * len_Bpe * lcd_sin))
        gamma = np.arccos(arg)
        along = Bpe * (lcd_sin * np.cos(gamma) / len_Bpe)[:, None]
        perp = np.cross(Bpe, cb) * (lcd_sin * np.sin(gamma)
                                    / (len_Bpe * np.linalg.norm(cb, axis=-1)))[:, None]
    p1 = e + along + perp
    p2 = e + along - perp

    if threshold is None:
        # The N-atom table is slightly biased towards one branch for short
        # remaining fragments (sample_states.cpp:258-268).
        threshold = 0.5
        if label == 0:
            threshold = 0.52 if rem <= 4 else (0.51 if rem <= 8 else 0.5)
    pick = rng().uniform(0.0, 1.0, n_states) <= threshold
    return np.where(pick[:, None], p1, p2)


def _sample_decrement(dist, label, rem, lcon, md0, md1):
    """
    Draw the end-to-end distance decrement from the empirical table.
    Builds the piecewise-linear conditional density over the 32 decrement bins,
    integrates it into a CDF (trapezoid rule, with partial first and last bins),
    and inverts it.  The bookkeeping matches the C++ line for line, including
    the truncated-toward-zero bin indices and the two edge cases where the
    reachable interval runs past the table.
    """
    S = md0.shape[0]
    table = dist.eted[label][rem] # (32, 32)
    distby = dist.dis_by[label][rem]
    delby = dist.del_by[label][rem]
    mindeld = dist.min_del[label][rem]

    x = lcon - dist.min_dis[label][rem]
    conlower = (x / distby + 0.00001).astype(np.int64)
    conlower = np.clip(conlower, 0, N_DIST_BIN - 2)
    x1 = conlower * distby
    x2 = x1 + distby
    w2 = (x2 - x)[:, None]
    w1 = (x - x1)[:, None]

    lo_row = table[conlower] # (S, 32)
    hi_row = table[conlower + 1]
    # Density linearly interpolated in the end-to-end distance direction.
    pdf_i = (w2 * lo_row + w1 * hi_row) / distby

    idx = np.arange(S)
    # int() in C truncates toward zero, and the reference multiplies by the
    # reciprocal rather than dividing; both are reproduced exactly.
    inv_delby = 1.0 / delby
    lowY = np.trunc((md0 - mindeld) * inv_delby).astype(np.int64)
    highY = np.trunc((md1 - mindeld) * inv_delby).astype(np.int64) + 1
    low_edge = (md0 - mindeld) <= 0
    high_edge = highY > N_DIST_BIN - 1
    lowY = np.where(low_edge, 0, np.clip(lowY, 0, N_DIST_BIN - 2))
    highY = np.where(high_edge, N_DIST_BIN - 1, np.clip(highY, 1, N_DIST_BIN - 1))

    pdf = np.zeros((S, N_DIST_BIN))
    denom = distby * delby

    # First (partial) bin.
    frac_lo = md0 - mindeld - delby * lowY
    bilinear_lo = ((lo_row[idx, lowY] * w2[:, 0] + hi_row[idx, lowY] * w1[:, 0])
                   * (delby - frac_lo)
                   + (lo_row[idx, lowY + 1] * w2[:, 0]
                      + hi_row[idx, lowY + 1] * w1[:, 0]) * frac_lo) / denom
    pdf[idx, lowY] = np.where(low_edge, pdf_i[idx, lowY], bilinear_lo)
    lowgap = np.where(low_edge, 0.0, delby - frac_lo)

    # Last (partial) bin.
    frac_hi = md1 - mindeld - delby * (highY - 1)
    bilinear_hi = ((lo_row[idx, highY - 1] * w2[:, 0]
                    + hi_row[idx, highY - 1] * w1[:, 0]) * (delby - frac_hi)
                   + (lo_row[idx, highY] * w2[:, 0]
                      + hi_row[idx, highY] * w1[:, 0]) * frac_hi) / denom
    pdf[idx, highY] = np.where(high_edge, pdf_i[idx, highY], bilinear_hi)
    highgap = np.where(high_edge, 0.0, frac_hi)

    # Interior bins, including the one just above the first partial bin, which
    # the C++ overwrites after the edge assignments.
    floor_ = np.minimum(lowY + 1, N_DIST_BIN - 1)
    interior = (np.arange(N_DIST_BIN)[None, :] >= floor_[:, None])
    interior &= (np.arange(N_DIST_BIN)[None, :] < highY[:, None])
    pdf = np.where(interior, pdf_i, pdf)
    pdf[idx, floor_] = pdf_i[idx, floor_]

    # Trapezoidal CDF over [lowY, highY].
    inc = np.zeros((S, N_DIST_BIN))
    mid = (np.arange(N_DIST_BIN)[None, :] > floor_[:, None])
    mid &= (np.arange(N_DIST_BIN)[None, :] < highY[:, None])
    inc[:, 1:] = np.where(mid[:, 1:], (pdf[:, 1:] + pdf[:, :-1]) * 0.5 * delby, 0.0)
    inc[idx, floor_] = (pdf[idx, floor_] + pdf[idx, lowY]) * 0.5 * lowgap
    cdf = np.cumsum(inc, axis=1)
    cdf[idx, highY] = (cdf[idx, np.maximum(highY - 1, 0)]
                       + (pdf[idx, highY] + pdf[idx, highY - 1]) * 0.5 * highgap)
    # Flat beyond the last bin so the inverse lookup cannot run past it.
    beyond = np.arange(N_DIST_BIN)[None, :] > highY[:, None]
    cdf = np.where(beyond, cdf[idx, highY][:, None], cdf)

    total = cdf[idx, highY]
    randr = rng().uniform(0.0, 1.0, S) * total

    # First bin whose cumulative mass reaches the draw.
    hit = np.argmax(cdf >= randr[:, None], axis=1)
    hit = np.clip(hit, 0, N_DIST_BIN - 1)
    prev = np.maximum(hit - 1, 0)

    with np.errstate(invalid="ignore", divide="ignore"):
        y_first = md0 + randr * lowgap / cdf[idx, floor_]
        span = cdf[idx, hit] - cdf[idx, prev]
        y_mid = (mindeld + delby * hit - delby
                 + (randr - cdf[idx, prev]) * delby / span)
        y_last = (mindeld + delby * (highY - 1)
                  + (randr - cdf[idx, np.maximum(highY - 1, 0)]) * highgap
                  / (total - cdf[idx, np.maximum(highY - 1, 0)]))

    y = np.where(hit <= floor_, y_first, np.where(hit >= highY, y_last, y_mid))
    # A zero-mass interval leaves nothing to sample; propagate as NaN so the
    # caller drops the state, which is what the C++ isnan() checks catch.
    return np.where(total > 0, y, np.nan)


# ---------------------------------------------------------------------------
# torsion-angle samplers
# ---------------------------------------------------------------------------

def sample_bb_angles(res_type, num_states, bb_type, native=None):
    """
    Backbone phi/psi/omega proposals, ported from sample_states.cpp.
    Only types 3 and 4 are implemented in the C++ code.
    3: perturb a native conformation. 4: uniform sampling
    types 1 and 2 are unfinished SQL stubs.  The production command line 
    uses -nds only, so no angle states are requested and this is never called.
    """
    # perturb native angles
    if bb_type == 3:
        if native is None:
            raise ValueError("bb_type 3 needs the native angles")
        out = np.empty((num_states, 3))
        for k in range(3):
            out[:, k] = box_muller_batch(native[k], 20, num_states)
        return out
    # random sample of angles 
    if bb_type == 4:
        out = np.empty((num_states, 3))
        out[:, 0] = frand(-PI, PI, num_states)
        out[:, 1] = frand(-PI, PI, num_states)
        out[:, 2] = PI
        return out
    # stub, raise error
    raise NotImplementedError(
        f"backbone angle sampling type {bb_type} is an unimplemented stub in "
        "the reference C++ (it issues SQL against a database that is not "
        "shipped); use type 3 or 4")


def sample_sc_angles(res_type, num_states, sc_type=2, native_chi=None):
    """
    Side-chain chi proposals ported from sample_states.cpp
    sc_type 1 or 2 draws a rotamer bin from the SCT_PF.txt distribution and
    then a uniform angle within the bin. type 3 perturbs the native chi angles
    by +-10 degrees.
    """
    # get number of rotamers. if 0, return empty list of sc angles
    num_rot = Rotamer.numRotBond[res_type]
    angles = np.zeros((num_states, 6))
    if num_rot == 0:
        return angles
    # use SCT_PF distribution.
    if sc_type in (1, 2):
        draws = rng().uniform(0.0, 1.0, num_states)
        for i, r in enumerate(draws):
            key = SCR.sample_key(res_type, r)
            if key is None:
                raise ValueError(f"no side-chain torsion data for type {res_type}")
            for j in range(num_rot - 1, -1, -1):
                angles[i, j] = ((key % 100) * SC_T_INT
                                + rng().uniform(0.0, SC_T_INT) - 180) * PI / 180
                key //= 100
        return angles
    # randomly perturb angles by 10 degrees
    if sc_type == 3:
        if native_chi is None:
            raise ValueError("sc_type 3 needs the native chi angles")
        for j in range(num_rot):
            r = rng().uniform(native_chi[j] - 10, native_chi[j] + 10, num_states)
            r = np.where(r < -180, r + 360, np.where(r > 180, r - 360, r))
            angles[:, j] = r * PI / 180
        return angles
    # if not a valid sc_type, raise error
    raise ValueError(f"unknown side-chain sampling type {sc_type}")
