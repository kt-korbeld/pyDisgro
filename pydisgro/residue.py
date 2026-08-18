import math
import os
from dataclasses import dataclass
import numpy as np

from .constants import *
from .geom import Atom, calCo, padded_array

# ---------------------------------------------------------------------------
# Residue class, including residue parameters loaded in from atomProp2.txt
# ---------------------------------------------------------------------------

class Residue:
    """
    One amino acid: a fixed-length list of Atoms depending on residue type
    plus a set of virtual atoms representing various centres.
    _atom is indexed by expected atoms based on the residue 
    template given in by atomProp2.txt, even when atoms are missing. 
    slot 0 to 5 are always N, CA, C, O, H, CB. 
    Absent atoms are Atom() placeholders with UNDEF atom type. 
    """

    # Class variables loaded in by InitPar/InitMap.
    Name1 = [] # 1 letter amino acid names
    Name3 = [] # 3 letter amino acid names
    FunctionalGroupAtoms = [] # atoms per residue that make up the functional group 
    cType = []
    vdwType = [] # VdW type per atom
    prev_atom = []
    bond_length = [] # bond lengths for each residue
    bond_angle = [] # radians, as in the C++
    torsion = [] # radians, -1234 marks a rotatable chi
    size = [] # residue size per residue, index map to D_RES
    sc_size = [] # side chain size per residue, index maps to D_RES
    bb_size = 1.93
    numAtom = []
    funcAtom = []

    AAMap = {}           # name conversions between 1/3-letter and full names
    AIMap = {}           # amino acid name -> integer type
    SIMap = {"H": 0, "E": 1, "C": 2}
    AtomMap = {}         # map atomname to index within the residue
    AtomIndexMap = {}    # map atomname to global atom type index

    def __init__(self, atoms):
        self._atom = atoms
        for i, atom in enumerate(self._atom):
            atom._parent = self
            atom._posn = i
        self._type = -1
        self._posn = 0
        self._chainName = ""
        self._ss = 0
        self._phi = 0.0
        self._psi = 0.0
        self._omega = 0.0
        self._scChi = np.zeros(5)
        self._scState = -1
        self._pdbIndex = 0      # index in pdb
        self._rotE = {}         # rotational energy
        self._parent = None     # parent residue to which atom belongs
        self._bbc = Atom()      # backbone centre (including CB)
        self._scc = Atom()      # side-chain centre
        self._center = Atom()   # whole-residue centre
        self._SC = Atom()       # pseudo side-chain atom used during growth
        self._FG = Atom()       # functional group centre
        self._res_adj = set()
        self._bb_adj = set()
        self._sc_adj = set()

    # basic properties
    @property
    def xyz(self):
        return np.stack([a.xyz for a in self._atom])
    @property
    def numAtomPresent(self):
        return len(self._atom)
    @property
    def name(self):
        return Residue.Name3[self._type] if self._type >= 0 else "UNK"
    def __repr__(self):
        return f"Residue({self.name} posn={self._posn} natom={len(self._atom)})"
    def copy(self):
        """
        Deep-ish copy: new Atom objects, shared class-level parameters.
        """
        atoms = []
        for a in self._atom:
            b = Atom(a.xyz[0], a.xyz[1], a.xyz[2], a._type)
            b._name = a._name
            b._Bfactor = a._Bfactor
            atoms.append(b)
        out = Residue(atoms)
        out._type = self._type
        out._posn = self._posn
        out._chainName = self._chainName
        out._pdbIndex = self._pdbIndex
        out._phi, out._psi, out._omega = self._phi, self._psi, self._omega
        out._scChi = self._scChi.copy()
        out._parent = self._parent
        for attr in ("_bbc", "_scc", "_center", "_SC", "_FG"):
            src = getattr(self, attr)
            dst = Atom(src.xyz[0], src.xyz[1], src.xyz[2], src._type)
            setattr(out, attr, dst)
        return out

    # centres 
    def _mean_of(self, atoms):
        """
        Calculate mean position of a set of atoms,
        excluding undefined atomtypes
        """
        pts = [a.xyz for a in atoms if a._type != UNDEF]
        if not pts:
            return None
        return np.mean(np.asarray(pts), axis=0)
    
    def cal_cent(self):
        """
        Centre over every defined atom of the residue.
        """
        avg = self._mean_of(self._atom)
        if avg is None:
            self._center = Atom()
        else:
            self._center = Atom.from_array(avg)
            self._center._type = 0
    
    def cal_bbc(self):
        """
        Calculate backbone centre 
        by taking first NUM_BB_ATOM slots.
        """
        avg = self._mean_of(self._atom[:NUM_BB_ATOM])
        if avg is None:
            self._bbc = Atom()
        else:
            self._bbc = Atom.from_array(avg)
            self._bbc._type = 0
    
    def cal_scc(self):
        """
        Calculate side-chain centre based on
        by taking everything after NUM_BB_ATOM slots.
        """
        avg = self._mean_of(self._atom[NUM_BB_ATOM:])
        if avg is None:
            self._scc = Atom()
        else:
            self._scc = Atom.from_array(avg)
            self._scc._type = 0

    def cal_fg(self):
        """
        Calculate centre of the side-chain functional group.
        includes some exceptions for different residues.
        """
        if self._type < 0 or self._type >= NUM_RES_TP or self._type == GLY:
            return
        if self._type == ALA:
            self._FG = self._atom[D_AT["ATM_CB"]]
            return
        acc = np.zeros(3)
        count = 0
        for i, at in enumerate(self._atom):
            if at._name in ("N", "CA", "C", "O", "CB", "H"):
                continue
            if at._type == UNDEF or at._type >= H_ATOM_TYPE:
                continue
            # The first atoms of long flexible side chains are treated as
            # backbone-like and excluded from the functional group.
            if self._type == 8 and i in (6, 7, 8):      # LYS
                continue
            if self._type == 14 and i in (6, 7):        # ARG
                continue
            if self._type == 16 and i == 7:             # THR
                continue
            acc += at.xyz
            count += 1
        tmp = Atom.from_array(acc / count) if count else Atom(0, 0, 0)
        tmp._type = 5
        self._FG = tmp

    def cal_sc(self):
        """
        Calculate the pseudo side-chain atom 
        for the _SC attribute
        """
        if self._type < 0 or self._type >= NUM_RES_TP or self._type == GLY:
            return
        if self._type == ALA:
            self._SC = self._atom[D_AT["ATM_CB"]]
            self._SC._name = "SC"
            return
        prev = [self._atom[D_AT["ATM_N"]],
                self._atom[D_AT["ATM_C"]],
                self._atom[D_AT["ATM_CA"]]]
        if all(p.is_origin() for p in prev):
            return
        sc = calCo(prev,
                   Atom.R_SC[self._type],
                   Residue.bond_angle[self._type][D_AT["ATM_CB"]],
                   math.pi * 122.55 / 180)
        sc._type = 5
        sc._name = "SC"
        self._SC = sc

    def AllHBond(self, given=True, taken=True):
        rst = []
        for a in self._atom:
            if given:
                rst.extend(HBond(a, g) for g in a.HGiven)
            if taken:
                rst.extend(HBond(t, a) for t in a.HTaken)
        return rst

    # parameter loading
    @classmethod
    def ResAtomMap(cls, res_type, inv=False):
        """
        get index in residue type based on the global atom names, 
        which are based on 1-letter res name, + atom name. 
        For example, KCA for Ca in lysine
        """
        if isinstance(res_type, str):
            if res_type not in cls.AAMap:
                return {}
            letter = cls.AAMap[res_type] if len(res_type) != 1 else res_type
        else:
            letter = cls.Name1[res_type]
        out = {a: cls.AtomMap[a] for a in cls.AtomMap if a[0] == letter}
        if inv:
            out = {v: k for k, v in out.items()}
        return out

    @classmethod
    def InitMap(cls):
        """
        Build the amino-acid naming maps.
        corresponds to residue.cpp InitMap.
        """
        AAMap_3 = {"Ala": "A", "Cys": "C", "Asp": "D", "Glu": "E", "Phe": "F",
                   "Gly": "G", "His": "H", "Ile": "I", "Lys": "K", "Leu": "L",
                   "Met": "M", "Asn": "N", "Pro": "P", "Gln": "Q", "Arg": "R",
                   "Ser": "S", "Thr": "T", "Val": "V", "Trp": "W", "Tyr": "Y",
                   "  C": "c", "  T": "t", "  U": "u", "  A": "a", "  G": "g",
                   "Unk": "U"}
        AAMap_full = {"Alanine": "A", "Cysteine": "C", "Aspartate": "D",
                      "Glutamate": "E", "Phenylalanine": "F", "Glycine": "G",
                      "Histidine": "H", "Isoleucine": "I", "Lysine": "K",
                      "Leucine": "L", "Methionine": "M", "Asparagine": "N",
                      "Proline": "P", "Glutamine": "Q", "Arginine": "R",
                      "Serine": "S", "Threonine": "T", "Valine": "V",
                      "Tryptophan": "W", "Tyrosine": "Y", "Disulfide": "Z",
                      "Unknown": "U", "Cytosine": "c", "Thymine": "t",
                      "Adenine": "a", "Guanine": "g", "Uracil": "u"}
        AAMap_caps = {k.upper(): v for k, v in AAMap_3.items()}
        AAMap_caps_inv = {v: k for k, v in AAMap_caps.items()}
        cls.AAMap = AAMap_3 | AAMap_caps | AAMap_caps_inv | AAMap_full

        AIMap_1 = {n: i for i, n in enumerate(AAMap_3.values()) if n != 'U'}
        AIMap_1["Z"], AIMap_1["U"] = 25, 26
        AIMap_3 = {cls.AAMap[k]: v for k, v in AIMap_1.items() if k in cls.AAMap}
        AIMap_inv = {v: k for k, v in AIMap_3.items()}
        cls.AIMap = AIMap_1 | AIMap_3 | AIMap_inv

    @classmethod
    def InitPar(cls, parFile=None, verbose=False):
        """
        Load residue templates, bond geometry and radii from atomProp2.txt.
        Bond angles and torsions are stored in radians
        so they can be fed straight to the calCo function.
        """
        if parFile is None:
            parFile = data_path(FILE_ATOMPROP)
        data_types = ["restype", "atomDependency", "bondParameter",
                      "residueSize", "torsionAngle"]
        if verbose:
            print("Reading Residue Parameters in ", parFile)
        if not os.path.exists(parFile):
            raise FileNotFoundError(f"cannot open parameter file {parFile}")

        listvars = ["numAtom", "Name1", "Name3", "cType", "vdwType",
                    "prev_atom", "bond_length", "bond_angle", "size",
                    "sc_size", "torsion"]
        for name in listvars:
            setattr(cls, name, [])
        cls.AtomMap, cls.AtomIndexMap = {}, {}
        atom_index = 0

        with open(parFile) as f:
            numLine, datatype = 0, ""
            for line in f:
                if line.startswith("#"):
                    match = [dt for dt in data_types if dt in line]
                    if not match:
                        datatype = ""
                        continue
                    datatype = match[0]
                    numLine = int([i for i in line.split() if i.isdigit()][0])
                    if verbose:
                        print(f"Reading in {numLine} {datatype} types")
                    continue
                if numLine <= 0:
                    continue

                tokens = line.split()
                if datatype == "restype":
                    if tokens[0] not in cls.AAMap:
                        raise ValueError("restype error, wrong AA type")
                    cls.numAtom.append((len(tokens) - 2) // 2)
                    cls.Name1.append(tokens[0])
                    cls.Name3.append(cls.AAMap[tokens[0]])
                    cls.cType.append(list(tokens[2::2]))
                    cls.vdwType.append([int(v) for v in tokens[3::2]])
                    for slot, atom in enumerate(tokens[2::2]):
                        cls.AtomMap[tokens[0] + atom] = slot
                        cls.AtomIndexMap[tokens[0] + atom] = atom_index
                        atom_index += 1
                    numLine -= 1
                elif datatype == "atomDependency":
                    if tokens[1] not in cls.AAMap:
                        raise ValueError("atomDependency error, wrong AA type")
                    p1s, p2s, p3s = tokens[3::4], tokens[4::4], tokens[5::4]
                    cls.prev_atom.append([[int(a), int(b), int(c)]
                                          for a, b, c in zip(p1s, p2s, p3s)])
                    numLine -= 1
                elif datatype == "bondParameter":
                    if tokens[1] not in cls.AAMap:
                        raise ValueError("bondParameter error, wrong AA type")
                    cls.bond_length.append([float(v) for v in tokens[3::3]])
                    cls.bond_angle.append([(math.pi / 180) * float(v)
                                           for v in tokens[4::3]])
                    numLine -= 1
                elif datatype == "residueSize":
                    if tokens[1] not in cls.AAMap:
                        raise ValueError("residueSize error, wrong AA type")
                    cls.size.append(float(tokens[2]))
                    cls.sc_size.append(float(tokens[4]))
                    numLine -= 1
                elif datatype == "torsionAngle":
                    if tokens[0] not in cls.AAMap:
                        raise ValueError("torsionAngle error, wrong AA type")
                    tor = []
                    for t in tokens[3::2]:
                        # -1234 marks a rotatable chi angle, sampled at runtime
                        tor.append(float(t) if t == "-1234"
                                   else (math.pi / 180) * float(t))
                    cls.torsion.append(tor)
                    numLine -= 1

        for name in listvars:
            value = getattr(cls, name)
            if name in ("cType", "Name1", "Name3"):
                setattr(cls, name, padded_array(value, fill_value="", dtype=str))
            elif name in ("vdwType", "prev_atom", "numAtom"):
                setattr(cls, name, padded_array(value, fill_value=-1234, dtype=int))
            else:
                setattr(cls, name, padded_array(value, fill_value=-1234, dtype=float))

# ---------------------------------------------------------------------------
# Data classes for rotamers and hydrogen bonds
# ---------------------------------------------------------------------------

class Rotamer:
    """
    One side-chain rotamer.
    """
    # Number of rotatable bonds per residue type (rotamer.cpp).
    numRotBond = [0, 1, 2, 3, 2,   # A C D E F
                  0, 2, 2, 4, 2,   # G H I K L
                  3, 2, 2, 3, 4,   # M N P Q R
                  1, 1, 1, 2, 2]   # S T V W Y
    def __init__(self, n=None):
        self.aaType = n if n is not None else 0
        self.intRep = 0
        self.prob = 0.0
        self.chi = []

class RotBond:
    """
    A rotatable bond: its angle and the width of the bin it came from.
    """
    __slots__ = ("angle", "range")
    def __init__(self, angle=0.0, rng=0.0):
        self.angle = angle
        self.range = rng

@dataclass
class HBond:
    """
    Hydrogen bond object
    """
    donor: Atom
    acceptor: Atom

@dataclass
class PNT_EN:
    """
    Interaction energy of a rotamer, used during side-chain modelling.
    """
    rotP: float   # residue position * 100 + rotamer index
    energy: float

@dataclass
class ResAtomIdxPair:
    """
    included in residue.h of original code,
    included for legacy, currently not in use.
    """
    ResIdx: int
    AtmIdx: int

@dataclass
class SCT:
    """
    Side-chain torsion data class.
    """
    torsion: list
    bfac: list

class SCR:
    """
    Side-chain torsion-angle distributions read from SCT_PF.txt.
    For each residue type we keep a cumulative probability and 
    the matching packed rotamer keys. this allows for a continious 
    mapping of probability to rotameric state. 
    replaces the C++ ``map<double,int>::upper_bound`` lookup.
    """

    CumProbs = []
    RotKeys = []

    @classmethod
    def sample_key(cls, res_type, r):
        """
        Return the rotamer key whose cumulative probability first exceeds r.
        """
        probs = cls.CumProbs[res_type]
        if len(probs) == 0:
            return None
        idx = int(np.searchsorted(probs, r, side="right"))
        if idx >= len(probs):
            idx = len(probs) - 1
        return int(cls.RotKeys[res_type][idx])

    @classmethod
    def InitSCAng(cls, scFile=None, verbose=False):
        """
        Load the side-chain torsion angle library.
        """
        if scFile is None:
            scFile = data_path(FILE_SCTORSION2)
        if verbose:
            print("Reading Side Chain Torsion Angle Parameters in ", scFile)
        if not os.path.exists(scFile):
            raise FileNotFoundError(f"cannot open parameter file {scFile}")

        keys = [[] for _ in range(NUM_RES_TP)]
        counts = [[] for _ in range(NUM_RES_TP)]
        total = np.zeros(NUM_RES_TP)

        with open(scFile) as f:
            for line in f:
                if not line or line[0] == "#":
                    continue
                tok = line.split()
                if not tok:
                    continue
                aa = int(tok[0])
                numRot = Rotamer.numRotBond[aa]
                if len(tok) < numRot + 3:
                    continue
                key = 0
                for j in range(1, numRot + 1):
                    key = key * 100 + int(tok[j])
                count = float(tok[numRot + 2])
                keys[aa].append(key)
                counts[aa].append(count)
                total[aa] += count

        cum_probs, rot_keys = [], []
        for i in range(NUM_RES_TP):
            if not keys[i]:
                cum_probs.append(np.zeros(0))
                rot_keys.append(np.zeros(0, dtype=np.int64))
                continue
            probs = np.asarray(counts[i]) / total[i]
            cum = np.cumsum(probs)
            k = np.asarray(keys[i], dtype=np.int64)
            # Guard against the last cumulative probability falling just short
            # of 1 because of floating point.
            if cum[-1] < 1.0:
                cum = np.append(cum, 1.0)
                k = np.append(k, k[-1])
            cum_probs.append(cum)
            rot_keys.append(k)

        cls.CumProbs = cum_probs
        cls.RotKeys = rot_keys

class RotLib:
    """
    Stub for the Richardson/Dunbrack rotamer libraries.
    DiSGro only uses the SCT_PF.txt torsion distributions in the SCR class. 
    the rotamer library files are not shipped and the code path is never taken.
    """
    scLib = 0
    RLR = []
    RLD = []
