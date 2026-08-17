# basic point and atom classes, rng, and mathematical operations

import os
import math
import numpy as np
from .constants import PI, UNDEF, FILE_ATOMPROP, data_path

# ---------------------------------------------------------------------------
# functions for random number generation
# ---------------------------------------------------------------------------

# set global rng object that consistently uses the same seed
_rng = np.random.default_rng() 

def seed(value=None):
    """
    seed the numpy random rng module
    """
    global _rng
    _rng = np.random.default_rng(value)
    return _rng

def rng():
    """
    load global rng object
    """
    return _rng

def frand(a, b, size=None):
    """
    match the C++ frand which samples 
    random floats between a and b.
    """
    return _rng.uniform(a, b, size)

def intrand(a, b, size=None):
    """
    match the C++ intrand which samples 
    random integers between a and b.
    """
    return _rng.integers(a, b, size)

def SampleOne(prob):
    """Pick an index with probability proportional to ``prob`` (util.cpp).

    Equivalent to the C++ linear scan against a uniform draw in [0, sum), but
    done with a cumulative sum so it is O(n) in numpy rather than in Python.
    """
    prob = np.asarray(prob, dtype=float)
    cum = np.cumsum(prob)
    total = cum[-1]
    if not total > 0:
        # The callers guard against this, but fall back to uniform rather than
        # aborting the way the C++ does.
        return int(_rng.integers(0, len(prob)))
    return int(np.searchsorted(cum, _rng.uniform(0.0, total), side="right"))


def box_muller_single(mean, sigma_deg):
    """Normal draw about *mean* with *sigma_deg* in degrees (util.cpp:927).

    The C++ folds values above pi back by subtracting the mean twice; that
    quirk is reproduced because it shapes the omega distribution.
    """
    sigma = sigma_deg * PI / 180.0
    u = _rng.uniform(0.0, 1.0)
    v = _rng.uniform(0.0, 1.0)
    value = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * PI * v) * sigma + mean
    if value > PI:
        value = (value - mean) - mean
    return value


def box_muller_batch(mean, sigma_deg, size):
    """Vectorised :func:`box_muller_single`."""
    sigma = sigma_deg * PI / 180.0
    u = _rng.uniform(0.0, 1.0, size)
    v = _rng.uniform(0.0, 1.0, size)
    value = np.sqrt(-2.0 * np.log(u)) * np.cos(2.0 * PI * v) * sigma + mean
    return np.where(value > PI, (value - mean) - mean, value)

# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------

def padded_array(data, fill_value=0, dtype=None):
    """
    Convert an arbitrarily nested list into a rectangular NumPy array,
    padding shorter lists with the set fill_value.
    """

    def shape(x):
        if not isinstance(x, list):
            return ()
        if not x:
            return (0,)
        child_shapes = [shape(c) for c in x]
        max_rank = max(len(s) for s in child_shapes)
        padded = [s + (0,) * (max_rank - len(s)) for s in child_shapes]
        return (len(x),) + tuple(max(vals) for vals in zip(*padded))

    def fill(arr, x, idx=()):
        if not isinstance(x, list):
            arr[idx] = x
            return
        for i, item in enumerate(x):
            fill(arr, item, idx + (i,))

    def leaf_values(x):
        if isinstance(x, list):
            for item in x:
                yield from leaf_values(item)
        else:
            yield x

    shp = shape(data)
    if dtype is None:
        values = list(leaf_values(data))
        dtype = np.result_type(*values) if values else np.asarray(fill_value).dtype
    elif dtype is str:
        maxlen = max((len(v) for v in leaf_values(data) if isinstance(v, str)), default=1)
        dtype = f"<U{max(maxlen, len(str(fill_value)))}"
    arr = np.full(shp, fill_value, dtype=dtype)
    fill(arr, data)
    return arr


def pairwise_distances(X, Y):
    """
    Euclidean distance matrix between X and Y, which should 
    be 3D-coordinates given as (n, 3) arrays
    """
    x2 = np.sum(X ** 2, axis=1)[:, None]
    y2 = np.sum(Y ** 2, axis=1)
    sq = x2 - 2.0 * (X @ Y.T) + y2
    return np.sqrt(np.maximum(sq, 0.0))


# ---------------------------------------------------------------------------
# functions for calculating new coordinates
# ---------------------------------------------------------------------------

def calCo_batch(a, b, c, length, bAngle, tAngle):
    """
    calculate coordinates for a given batch of input coordinates
    from length, angle, and torsion and 3 predecessor atoms a,b,c.
    For the chain a-b-c, return the position of d such that a-b-c-d
    with bond length c-d set by variable length, the angle b-c-d
    set by variable bAngle, and the torsion a-b-c-d set by variable tAngle.
    Angles are in radians. All arguments broadcast.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)

    svdv = a - b
    su = c - b
    u = su / np.linalg.norm(su, axis=-1, keepdims=True)
    # projection of a-b onto the b-c direction, removed to get the perpendicular
    d = np.sum(svdv * u, axis=-1, keepdims=True)
    perp = svdv - u * d
    v = perp / np.linalg.norm(perp, axis=-1, keepdims=True)
    w = np.cross(u, v)

    length = np.asarray(length, dtype=float)[..., None]
    bAngle = np.asarray(bAngle, dtype=float)[..., None]
    tAngle = np.asarray(tAngle, dtype=float)[..., None]
    sin_b = np.sin(PI - bAngle)
    return (c
            + u * length * np.cos(PI - bAngle)
            + v * length * sin_b * np.cos(tAngle)
            + w * length * sin_b * np.sin(tAngle))

def angle_batch(a, b, c):
    """
    Bond angle a-b-c in degrees
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d12 = np.linalg.norm(a - b, axis=-1)
    d23 = np.linalg.norm(c - b, axis=-1)
    dot = np.sum((a - b) * (b - c), axis=-1)
    ct = np.clip(dot / (d12 * d23), -1.0, 1.0)
    return 180.0 - np.degrees(np.arccos(ct))

def torsion_batch(a, b, c, d):
    """
    Torsion angle a-b-c-d in degrees
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)

    ij = a - b
    kj = c - b
    kl = c - d

    di = np.cross(ij, kj)     # normal to plane 1
    gi = np.cross(kl, kj)     # normal to plane 2 (sign as in the C++)

    bi = np.linalg.norm(di, axis=-1)
    bk = np.linalg.norm(gi, axis=-1)
    ct = np.clip(np.sum(di * gi, axis=-1) / (bi * bk), -1.0, 1.0)
    ap = np.arccos(ct)

    s = np.sum(kj * np.cross(gi, di), axis=-1)
    ap = np.where(s < 0.0, -ap, ap)
    ap = np.where(ap > 0.0, PI - ap, -(PI + ap))
    return np.degrees(ap)


# ---------------------------------------------------------------------------
# Point and Atom classes
# ---------------------------------------------------------------------------

class Point:
    """
    A 3D point with the arithmetic the C++ ``Point`` class provides.
    multiplication by another point is the cross product, matching the C++
    the dot product has been repaced with dot from the C++ ^ operator
    """
    __slots__ = ("xyz",)

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.xyz = np.asarray((x, y, z), dtype=float)

    # interact with data from/to arrays
    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2])
    @staticmethod
    def _asarray(x):
        return x.xyz if isinstance(x, Point) else np.asarray(x, dtype=float)

    # basics
    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Point.from_array(self.xyz + other)
        return Point.from_array(self.xyz + self._asarray(other))
    def __sub__(self, other):
        if isinstance(other, (float, int)):
            return Point.from_array(self.xyz - other)
        return Point.from_array(self.xyz - self._asarray(other))
    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Point.from_array(self.xyz / other)
        return Point.from_array(self.xyz / self._asarray(other))
    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Point.from_array(self.xyz * other)
        return Point.from_array(np.cross(self.xyz, self._asarray(other)))
    def sum(self):
        return float(np.sum(self.xyz))
    def square(self):
        return float(np.sum(self.xyz ** 2))
    def pabs(self):
        return float(np.linalg.norm(self.xyz))
    def dot(self, other):
        return float(self.xyz @ self._asarray(other))
    def dist(self, other):
        return float(np.linalg.norm(self.xyz - self._asarray(other)))
    def disquare(self, other):
        return float(np.sum((self.xyz - self._asarray(other)) ** 2))

    def angle(self, p1, p3):
        """
        Calculate angle p1-self-p3 in degrees
        """
        return float(angle_batch(self._asarray(p1), self.xyz, self._asarray(p3)))

    def torsion(self, p1, p2, p3):
        """
        Torsion p1-p2-p3-self in degrees (self is the *last* atom).
        """
        return float(torsion_batch(self._asarray(p1), self._asarray(p2),
                                   self._asarray(p3), self.xyz))

    def is_origin(self):
        """
        True when the point sits exactly at (0, 0, 0).
        The C++ uses this as an "unset coordinate" marker in several energy
        loops, so the exact comparison is intentional.
        """
        return bool(np.all(self.xyz == 0.0))
        
    def randomizeCo(self):
        """
        generate randomized coordinate
        """
        self.xyz = _rng.random(3)

class Atom(Point):
    """
    An atom: a point object plus force-field type, name and H-bond bookkeeping.
    """
    __slots__ = ("_type", "_name", "_Bfactor", "_posn", "_globalposn",
                 "_parent", "_state", "HGiven", "HTaken")

    # Static class variables, populated by InitPar from atomProp2.txt.
    vdw_adj = 1
    radius = np.zeros(1)
    welldepth = np.zeros(1)
    s_volume = np.zeros(1)
    s_lambda = np.zeros(1)
    s_dgfree = np.zeros(1)
    acceptor = np.zeros(1, dtype=int)
    donor = np.zeros(1, dtype=int)
    hbondH = np.zeros(1, dtype=int)
    # Pseudo side-chain atom radius per residue type.
    # likely added post-hoc in original code
    R_SC = np.array([1.53, 2.14, 2.49, 3.16, 3.42,
                     0.00, 3.17, 2.35, 3.57, 2.65,
                     3.00, 2.52, 1.88, 3.15, 4.17,
                     1.95, 1.95, 1.97, 3.89, 3.78,
                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def __init__(self, x=0.0, y=0.0, z=0.0, atom_type=UNDEF):
        super().__init__(x, y, z)
        self._type = atom_type
        self._name = ""
        self._Bfactor = 0.0
        self._posn = -1
        self._globalposn = -1
        self._parent = None
        self._state = 0
        self.HGiven = []
        self.HTaken = []

    @classmethod
    def from_array(cls, arr):
        return cls(arr[0], arr[1], arr[2])

    def __repr__(self):
        return f"Atom(name={self._name} type={self._type} pos={self.xyz})"

    def copy_pos(self, other):
        self.xyz = np.array(self._asarray(other), dtype=float)

    def reset(self):
        self._state = 0
        self.HGiven.clear()
        self.HTaken.clear()

    def any_hbond(self):
        return bool(self.HGiven or self.HTaken)

    def set_parent(self, residue):
        self._parent = residue

    def is_heavy(self):
        """
        check if heavy atom (i.e non-H or placeholder)
        """
        from .constants import H_ATOM_TYPE
        return (self._type != UNDEF and self._type < H_ATOM_TYPE
                and not self.is_origin())

    @classmethod
    def InitPar(cls, parFile=None, verbose=False):
        """
        Load per-atom-type force-field parameters from atomProp2.txt.
        """
        # do some basic checks
        if parFile is None:
            parFile = data_path(FILE_ATOMPROP)
        if verbose:
            print("Reading Atom Parameters in ", parFile)
        if not os.path.exists(parFile):
            raise FileNotFoundError(f"cannot open parameter file {parFile}")

        # initialize parameters to be loaded in
        radius, welldepth, s_volume = [0.0], [0.0], [0.0]
        s_lambda, s_dgfree = [0.0], [0.0]
        acceptor, donor, hbondH = [0], [0], [0]

        with open(parFile) as f:
            numLine, datatype = 0, ""
            for line in f:
                # line starting with # should contain how many lines to expect
                if line.startswith("#"):
                    if "atom_parameters" in line:
                        datatype = "atom_parameters"
                        numLine = int([i for i in line.split() if i.isdigit()][0])
                        if verbose:
                            print(f"Reading in {numLine} atom_parameters types")
                    else:
                        datatype = ""
                    continue
                # for the expected number of lines, load in atom_parameters
                if numLine > 0 and datatype == "atom_parameters":
                    linedata = line.split("#")[0].replace(",", " ").split()
                    radius.append(cls.vdw_adj * float(linedata[0]))
                    welldepth.append(float(linedata[1]))
                    s_volume.append(float(linedata[2]))
                    s_lambda.append(float(linedata[3]))
                    s_dgfree.append(float(linedata[4]))
                    acceptor.append(int(linedata[5]))
                    donor.append(int(linedata[6]))
                    hbondH.append(int(linedata[7]))
                    numLine -= 1

        # load data into attributes
        cls.radius = np.array(radius, dtype=float)
        cls.welldepth = np.array(welldepth, dtype=float)
        cls.s_volume = np.array(s_volume, dtype=float)
        cls.s_lambda = np.array(s_lambda, dtype=float)
        cls.s_dgfree = np.array(s_dgfree, dtype=float)
        cls.acceptor = np.array(acceptor, dtype=int)
        cls.donor = np.array(donor, dtype=int)
        cls.hbondH = np.array(hbondH, dtype=int)


def calCo(prev_atoms, length, bAngle, tAngle):
    """
    calculates new coordnates from length, angle, and torsion
    by calling the calCo batch function
    """
    pos = calCo_batch(Point._asarray(prev_atoms[0]),
                      Point._asarray(prev_atoms[1]),
                      Point._asarray(prev_atoms[2]),
                      length, bAngle, tAngle)
    return Atom.from_array(pos)
