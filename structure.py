# structure module, used to load and represent the entire protein structure

import math
import os

import numpy as np
from .constants import *
from .geom import Atom, angle_batch, calCo, calCo_batch, torsion_batch
from .residue import Residue

# ---------------------------------------------------------------------------
# flat-array view of Structure class
# ---------------------------------------------------------------------------

class flatStructure:
    """
    Version of the protein class with all data turned into unified 
    numpy arrays, to allow numpy to perform fast bulk calculations. 
    The original Structure class is kept for ease of use and readability.
    the .from_structure() and the .write_back() functions allow for 
    conversion between the flatStructure and Structure classes.
    """

    __slots__ = ("xyz", "atype", "ridx", "slot", "res_start", "res_type",
                 "res_natom", "res_center", "res_bbc", "res_scc", "res_size",
                 "res_sc_size", "numRes", "numAtom", "named")

    def __init__(self):
        self.xyz = None # array of coordinates
        self.atype = None # array of atom types
        self.ridx = None # res index
        self.slot = None
        self.named = None # atom names
        self.res_start = None # index of start residue
        self.res_type = None # residue types
        self.res_natom = None # number of atoms per residue
        self.res_center = None # center of the complete residue
        self.res_bbc = None # backbone center of residue
        self.res_scc = None # side chain center of the residue
        self.res_size = None # residue size
        self.res_sc_size = None # side chain size
        self.numRes = 0 # number of residues
        self.numAtom = 0 # number of atoms

    # construction
    @classmethod
    def from_structure(cls, struct):
        self = cls()
        residues = struct._res
        nres = len(residues) - 1 # _res[0] is the placeholder
        counts = [len(r._atom) for r in residues]
        starts = np.zeros(len(residues) + 1, dtype=np.int64)
        starts[1:] = np.cumsum(counts)
        total = int(starts[-1])

        self.numRes = nres
        self.numAtom = total
        self.xyz = np.zeros((total, 3))
        self.atype = np.full(total, UNDEF, dtype=np.int32)
        self.ridx = np.zeros(total, dtype=np.int32)
        self.slot = np.zeros(total, dtype=np.int32)
        self.named = np.zeros(total, dtype=bool)
        self.res_start = starts
        self.res_type = np.full(len(residues), -1, dtype=np.int32)
        self.res_natom = np.zeros(len(residues), dtype=np.int32)
        self.res_center = np.zeros((len(residues), 3))
        self.res_bbc = np.zeros((len(residues), 3))
        self.res_scc = np.zeros((len(residues), 3))

        for i, res in enumerate(residues):
            lo, hi = starts[i], starts[i + 1]
            if hi > lo:
                self.xyz[lo:hi] = np.stack([a.xyz for a in res._atom])
                self.atype[lo:hi] = [a._type for a in res._atom]
                self.named[lo:hi] = [a._name != "" for a in res._atom]
                self.ridx[lo:hi] = i
                self.slot[lo:hi] = np.arange(hi - lo)
            self.res_type[i] = res._type
            self.res_natom[i] = len(res._atom)
            self.res_center[i] = res._center.xyz
            self.res_bbc[i] = res._bbc.xyz
            self.res_scc[i] = res._scc.xyz

        rt = np.clip(self.res_type, 0, NUM_RES_TP - 1)
        self.res_size = np.where(self.res_type < 0, 0.0,
                                 np.asarray(Residue.size)[rt])
        self.res_sc_size = np.where(self.res_type < 0, 0.0,
                                    np.asarray(Residue.sc_size)[rt])
        return self

    def copy(self):
        """
        Copy the mutable parts. the immutable index arrays are shared.
        """
        out = flatStructure()
        out.xyz = self.xyz.copy()
        out.atype = self.atype.copy()
        out.res_center = self.res_center.copy()
        out.res_bbc = self.res_bbc.copy()
        out.res_scc = self.res_scc.copy()
        out.res_natom = self.res_natom.copy()
        # These never change during sampling.
        out.ridx = self.ridx
        out.slot = self.slot
        out.named = self.named
        out.res_start = self.res_start
        out.res_type = self.res_type
        out.res_size = self.res_size
        out.res_sc_size = self.res_sc_size
        out.numRes = self.numRes
        out.numAtom = self.numAtom
        return out

    # selections
    def index(self, res, slot):
        """
        Global atom index of selected atom slot in selected res.
        """
        return int(self.res_start[res]) + slot
    
    def atom(self, res, slot):
        """
        Coordinates of a selected atom slot in selected res
        """
        return self.xyz[int(self.res_start[res]) + slot]
    
    def residue_slice(self, res):
        """
        Coordinate slice for selected res.
        """
        return slice(int(self.res_start[res]), int(self.res_start[res + 1]))

    def heavy_mask(self):
        """
        Atoms that carry a real heavy-atom type and a real position.
        The all-zero coordinate test reproduces the C++ convention that an atom
        left at the origin is "not placed" (used for the missing loop atoms,
        which the input PDB labels as H at 0.000).
        """
        return ((self.atype != UNDEF) & (self.atype < H_ATOM_TYPE)
                & ~np.all(self.xyz == 0.0, axis=1))

    # functions for the centres
    def update_center(self, res, sidechain=False):
        """
        Recompute the centres of one residue, matches structure.cpp:SinglecalCenter.
        sidechain=False matches the C++ type == 1: backbone atoms only,
        and the residue centre equals the backbone centre.
        """
        lo = int(self.res_start[res])
        hi = int(self.res_start[res + 1])
        nat = int(self.res_natom[res])
        coords = self.xyz[lo:hi]
        types = self.atype[lo:hi]
        # SinglecalCenter skips types > 20, i.e. hydrogens and the placeholder.
        ok = (types != UNDEF) & (types <= 20) & ~np.all(coords == 0.0, axis=1)

        bb = ok.copy()
        bb[NUM_BB_ATOM:] = False
        n_bb = int(bb.sum())

        if sidechain:
            sc = ok.copy()
            sc[:NUM_BB_ATOM] = False
            sc[nat:] = False
            n_sc = int(sc.sum())
        else:
            n_sc = 0

        self.res_bbc[res] = coords[bb].sum(axis=0) / n_bb if n_bb else 0.0
        if sidechain:
            if n_sc:
                self.res_scc[res] = coords[sc].sum(axis=0) / n_sc
            elif self.res_type[res] == ALA:
                self.res_scc[res] = coords[ATM_CB]
            elif self.res_type[res] == GLY:
                self.res_scc[res] = coords[ATM_CA]
            total = n_bb + n_sc
            if total:
                acc = coords[bb].sum(axis=0)
                acc = acc + coords[sc].sum(axis=0)
                self.res_center[res] = acc / total
            else:
                self.res_center[res] = 0.0
        else:
            self.res_center[res] = self.res_bbc[res]

    def update_centers(self, start, end, sidechain=False):
        for i in range(start, end + 1):
            self.update_center(i, sidechain)

    def write_back(self, struct, start=1, end=None):
        """Copy coordinates and centres back into the object model."""
        if end is None:
            end = self.numRes
        for i in range(start, end + 1):
            lo = int(self.res_start[i])
            res = struct._res[i]
            for slot, atom in enumerate(res._atom):
                atom.xyz = self.xyz[lo + slot].copy()
                atom._type = int(self.atype[lo + slot])
            res._center = Atom.from_array(self.res_center[i])
            res._bbc = Atom.from_array(self.res_bbc[i])
            res._scc = Atom.from_array(self.res_scc[i])


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

class Structure:
    """
    A protein structure represented as a list of Residue objects.
    Closest to the actual C++ implementation, and mainly used for convenience.
    For fast bulk operations, the flatStructure class is used.
    """
    _T = 1.0
    _sequence = ""

    def __init__(self, residues=None):
        # _res[0] is a placeholder so that ind start at 1 and calCo has a frame for first res
        res0 = Residue([Atom(-1, 1, 1), Atom(0, 1, 1), Atom(0, 1, 0)])
        if residues:
            self._res = [res0] + list(residues)
        else:
            # if empty, include placholder residue
            self._res = [res0, Residue([Atom(0, 0, 0), Atom(1.458, 0, 0),
                                        Atom(0, -1.01, 0)])]
        for i, res in enumerate(self._res):
            res._parent = self
            res._posn = i

        self._numChain = 0
        self._firstResidues = []
        self._chainName = ""
        self._ProtName = ""
        self._weight = 0.0
        self._weight_2 = 0.0
        self._enArr = np.zeros(25)
        self._energy = 0.0
        self._scProb = 0.0
        self._useNatSC = True
        self.G_rmsd = 1e5
        self.allatm_rmsd = 1e5
        self.LoopTorsion = 0.0
        self.MEdis1 = 0.0
        self.MEdis2 = 0.0
        self._single_rmsd = []
        self._toBeSampled = []
        self.Closed = False
        self.Success = False
        self._ssPred = []
        self._ssPredProb = []
        self._saPred = []
        self._ATM_OXT = Atom()
        self._ATM_OXT._type = 20
        self._Confcenter = Atom()
        self.missSeq = []
        self.missSeqPos = []
        self.missSeqfrom1 = []

    # basics
    @property
    def numRes(self):
        return len(self._res) - 1
    @property
    def xyz(self):
        return np.vstack([r.xyz for r in self._res])
    @property
    def _atom(self):
        out = []
        for res in self._res:
            out.extend(res._atom)
        return out
    def __repr__(self):
        return f"Structure({self._ProtName!r}, numRes={self.numRes})"
    def copy(self):
        out = Structure([r.copy() for r in self._res[1:]])
        out._ProtName = self._ProtName
        out._chainName = self._chainName
        out._energy = self._energy
        out._enArr = self._enArr.copy()
        out._weight = self._weight
        out._weight_2 = self._weight_2
        out.Closed = self.Closed
        out.Success = self.Success
        return out
    def StoreSequence(self):
        seq = []
        for r in self._res[1:]:
            if 0 <= r._type < NUM_RES_TP:
                seq.append(Residue.Name1[r._type])
            else:
                seq.append("?")
        self._sequence = "".join(seq)

    def StoreGlobalposn(self):
        for i, a in enumerate(self._atom):
            a._globalposn = i

    def flatten(self):
        """
        Build the flat-array view of this structure.
        """
        return flatStructure.from_structure(self)

    # functions for the centres
    def calCenter(self, start, end, sidechain=True):
        """
        Recompute residue centres over from residue start to end
        """
        for i in range(start, end + 1):
            res = self._res[i]
            bb = np.zeros(3)
            sc = np.zeros(3)
            n_bb = n_sc = 0
            for j, at in enumerate(res._atom[:NUM_BB_ATOM]):
                if at._type > 20 or at._type == UNDEF or at.is_origin():
                    continue
                bb += at.xyz
                n_bb += 1
            if sidechain:
                for at in res._atom[NUM_BB_ATOM:]:
                    if at._type > 20 or at._type == UNDEF or at.is_origin():
                        continue
                    sc += at.xyz
                    n_sc += 1

            res._bbc = Atom.from_array(bb / n_bb) if n_bb else Atom()
            if sidechain:
                if n_sc:
                    res._scc = Atom.from_array(sc / n_sc)
                elif res._type == ALA:
                    res._scc = res._atom[ATM_CB]
                elif res._type == GLY:
                    res._scc = res._atom[ATM_CA]
                else:
                    res._scc = Atom()
            if n_bb + n_sc:
                if sidechain:
                    res._center = Atom.from_array((bb + sc) / (n_bb + n_sc))
                else:
                    res._center = Atom.from_array(res._bbc.xyz)
                res._center._type = 0
            else:
                res._center = Atom()

    def calSC(self, start, end):
        for i in range(start, end + 1):
            self._res[i].cal_sc()

    def fill_atom_names(self):
        """
        Give every atom slot its template name.
        Atoms absent from the input PDB keep an empty name.
        """
        for res in self._res[1:]:
            if not (0 <= res._type < NUM_RES_TP):
                continue
            inv = Residue.ResAtomMap(res._type, inv=True)
            for slot, atom in enumerate(res._atom):
                if not atom._name and slot in inv:
                    atom._name = inv[slot][1:]

    # calculate atom coordinates for new residues
    def calBBCo(self, resInd, res, phi, psi, omega):
        """
        Place the backbone atoms of residue at resInd
        res receives C, O, CB and the pseudo side-chain atom of residue resInd
        plus the N and CA of residue resInd + 1. 
        """
        prev = self._res[resInd - 1]
        cur = self._res[resInd]
        type_prev = cur._type                     # C, O, CB, SC
        type_next = self._res[resInd + 1]._type   # N, CA, H

        # C, from phi
        res._atom[ATM_C] = calCo([prev._atom[ATM_C], cur._atom[ATM_N],
                                  cur._atom[ATM_CA]],
                                 Residue.bond_length[type_prev][ATM_C],
                                 Residue.bond_angle[type_prev][ATM_C], phi)
        # O, from psi + pi
        res._atom[ATM_O] = calCo([cur._atom[ATM_N], cur._atom[ATM_CA],
                                  res._atom[ATM_C]],
                                 Residue.bond_length[type_prev][ATM_O],
                                 Residue.bond_angle[type_prev][ATM_O], psi + PI)
        # N of the next residue, from psi
        res._atom[ATM_N] = calCo([cur._atom[ATM_N], cur._atom[ATM_CA],
                                  res._atom[ATM_C]],
                                 Residue.bond_length[type_next][ATM_N],
                                 Residue.bond_angle[type_next][ATM_N], psi)
        # CA of the next residue, from omega
        res._atom[ATM_CA] = calCo([cur._atom[ATM_CA], res._atom[ATM_C],
                                   res._atom[ATM_N]],
                                  Residue.bond_length[type_next][ATM_CA],
                                  Residue.bond_angle[type_next][ATM_CA], omega)
        if cur._type != GLY:
            res._atom[ATM_CB] = calCo([cur._atom[ATM_N], res._atom[ATM_C],
                                       cur._atom[ATM_CA]],
                                      Residue.bond_length[type_prev][ATM_CB],
                                      Residue.bond_angle[type_prev][ATM_CB],
                                      PI * 122.55 / 180)
            res._SC = calCo([cur._atom[ATM_N], res._atom[ATM_C],
                             cur._atom[ATM_CA]],
                            Atom.R_SC[type_prev],
                            Residue.bond_angle[type_prev][ATM_CB],
                            PI * 122.55 / 180)
        return res

    def calSCCo(self, res, torAngles):
        """
        Build side-chain coordinates for res from specified chi angles.
        Port of Structure::calSCCo. update _scc and _center afterwards.
        """
        rt = res._type
        if rt == ALA:
            res._scc = res._atom[ATM_CB]
        elif rt == GLY:
            res._scc = res._atom[ATM_CA]
        else:
            acc = np.zeros(3)
            n_heavy = 0
            rot_count = 0
            n_slots = int(Residue.numAtom[rt])
            for j in range(NUM_BB_ATOM, n_slots):
                p = Residue.prev_atom[rt][j]
                tor = Residue.torsion[rt][j]
                if tor == -1234:
                    torsion_angle = torAngles[rot_count]
                    rot_count += 1
                else:
                    torsion_angle = tor
                res._atom[j].xyz = calCo_batch(res._atom[p[0]].xyz,
                                               res._atom[p[1]].xyz,
                                               res._atom[p[2]].xyz,
                                               Residue.bond_length[rt][j],
                                               Residue.bond_angle[rt][j],
                                               torsion_angle)
                if res._atom[j]._type < H_ATOM_TYPE:
                    acc += res._atom[j].xyz
                    n_heavy += 1
            if n_heavy:
                res._scc = Atom.from_array(acc / n_heavy)

        if rt in (ALA, GLY):
            res._center = Atom.from_array(res._bbc.xyz)
        else:
            n_heavy = max(n_heavy, 1)
            res._center = Atom.from_array(
                (res._bbc.xyz * NUM_BB_ATOM + res._scc.xyz * n_heavy)
                / (NUM_BB_ATOM + n_heavy))
        return res

    # closure test
    def IsClosed(self, End):
        """
        Whether the chain is geometrically closed at residue specified in End.
        Port of Structure::IsClosed: checks the two
        bond lengths and two bond angles that analytic closure can distort,
        plus the omega torsion into the following residue.
        """
        r_end = self._res[End]
        r_next = self._res[End + 1]
        # get distances between backbone atoms
        d1 = r_end._atom[ATM_N].dist(r_end._atom[ATM_CA])
        d2 = r_end._atom[ATM_CA].dist(r_end._atom[ATM_C])
        # get angles and torsion between backbone atoms
        a1 = float(angle_batch(r_end._atom[ATM_N].xyz, r_end._atom[ATM_CA].xyz,
                               r_end._atom[ATM_C].xyz))
        a2 = float(angle_batch(r_end._atom[ATM_CA].xyz, r_end._atom[ATM_C].xyz,
                               r_next._atom[ATM_N].xyz))
        t1 = float(torsion_batch(r_end._atom[ATM_CA].xyz, r_end._atom[ATM_C].xyz,
                                 r_next._atom[ATM_N].xyz, r_next._atom[ATM_CA].xyz))
        # check if all closure conditions are met
        d_N_CA_success = bool(CLOSED_N_CA_MIN < d1 < CLOSED_N_CA_MAX)
        d_CA_C_success = bool(CLOSED_CA_C_MIN < d2 < CLOSED_CA_C_MAX)
        ang_N_CA_C_success = bool(CLOSED_ANG_NCAC_MIN < a1 < CLOSED_ANG_NCAC_MAX)
        ang_CA_C_N_success = bool(CLOSED_ANG_CACN_MIN < a2 < CLOSED_ANG_CACN_MAX)
        tor_success = bool(abs(t1) > CLOSED_OMEGA_MIN)
        # return if all conditions are met
        return bool(d_N_CA_success and d_CA_C_success and ang_N_CA_C_success and ang_CA_C_N_success and tor_success)

    # load, prepare and save pdb
    @staticmethod
    def readPdb(input_pdb, SelRes=None):
        """
        Read a PDB file or list of lines into a Structure object.
        Missing loop atoms are expected to be labelled as H atoms with zero
        coordinates, as described in the DiSGro readme.
        they end up as placeholder atoms with _type == UNDEF.
        """
        if isinstance(input_pdb, str):
            if not os.path.exists(input_pdb):
                raise FileNotFoundError(f"cannot open input pdb {input_pdb}")
            with open(input_pdb) as f:
                pdb_lines = f.readlines()
        else:
            pdb_lines = list(input_pdb)

        pdb_lines = [l for l in pdb_lines if l[:6] == "ATOM  "]
        residues = []
        cur_atoms = None
        cur_res = cur_nr = cur_chain = None

        def flush():
            if cur_atoms is None:
                return
            res = Residue(cur_atoms)
            res._type = Residue.AIMap[cur_res]
            res._posn = len(residues) + 1
            res._pdbIndex = int(cur_nr)
            res._chainName = cur_chain
            residues.append(res)

        for line in pdb_lines:
            if len(line) < 54 or len(line) > 90:
                raise ValueError(f"malformed atom line in PDB:\n{line}")
            res, resnr, chain = line[17:20], line[22:26], line[21]
            # skip deuterium, alternate locations and insertion codes
            if line[13] == "D" or line[16] not in (" ", "A") or line[26] != " ":
                continue
            if res not in Residue.AAMap:
                continue
            if SelRes and (chain + res) not in SelRes:
                continue

            if (cur_res is None) or resnr != cur_nr or chain != cur_chain:
                flush()
                cur_atoms = [Atom() for _ in range(int(Residue.numAtom[Residue.AIMap[res]]))]
                cur_res, cur_nr, cur_chain = res, resnr, chain

            at_name = line[12:16].strip()
            if at_name == "OXT":
                continue
            key = Residue.Name1[Residue.AIMap[res]] + at_name
            if key not in Residue.AtomMap:
                continue                      # atom not in the residue template
            slot = Residue.AtomMap[key]
            restype = Residue.AIMap[res]

            atom = Atom(float(line[30:38]), float(line[38:46]), float(line[46:54]))
            atom._name = at_name
            atom._type = int(Residue.vdwType[restype][slot])
            atom._Bfactor = float(line[60:66]) if len(line) >= 66 else 0.0
            cur_atoms[slot] = atom

        flush()

        out = Structure(residues)
        # Every template slot gets its force-field type, whether or not the
        # atom was present in the file (structure.cpp:505-511).  Absent atoms
        # are then identified by their all-zero coordinates, and by an empty
        # ``_name``; both markers are relied on by the energy terms.
        for res in out._res[1:]:
            for slot, atom in enumerate(res._atom):
                atom._type = int(Residue.vdwType[res._type][slot])
                atom._posn = slot
            res.cal_fg()
        out.calCenter(1, out.numRes, sidechain=True)
        out.StoreSequence()
        out.StoreGlobalposn()
        if isinstance(input_pdb, str):
            out._ProtName = os.path.basename(input_pdb).rsplit(".", 1)[0]
        return out

    def addH(self):
        """
        Add missing hydrogens. makes a guess based on reasonable angles. 
        """
        prev_res = None
        for i, res in enumerate(self._res):
            if not (0 <= res._type < NUM_RES_TP):
                prev_res = res
                continue
            rt = res._type
            letter = Residue.Name1[rt]
            # Skip residues that are missing a heavy atom -- we cannot place
            # hydrogens off atoms that are not there.
            slot_map = Residue.ResAtomMap(rt)
            heavy_slots = [v for k, v in slot_map.items() if "H" not in k[1:]]
            if any(res._atom[s]._type == UNDEF for s in heavy_slots):
                prev_res = res
                continue

            if res._atom[ATM_H]._type == UNDEF:
                slot = Residue.AtomMap[letter + "H"]
                if i == 1:
                    # First residue: no preceding C, so use a -75 degree guess.
                    newH = calCo([res._atom[ATM_C], res._atom[ATM_CA],
                                  res._atom[ATM_N]], 1.01, PI * (-75) / 180, 0)
                elif rt == 12:            # PRO has no backbone H
                    newH = Atom.from_array(res._atom[ATM_N].xyz)
                else:
                    newH = calCo([prev_res._atom[ATM_CA], prev_res._atom[ATM_C],
                                  res._atom[ATM_N]], 1.01, PI * 123 / 180, 0)
                res._atom[slot].xyz = np.round(newH.xyz, 3)
                res._atom[slot]._name = "H"
                res._atom[slot]._type = 25

            inv_map = Residue.ResAtomMap(rt, inv=True)
            for j, at in enumerate(res._atom):
                if at._type != UNDEF:
                    continue
                p = Residue.prev_atom[rt][j]
                newH = calCo([res._atom[p[0]], res._atom[p[1]], res._atom[p[2]]],
                             1.01, Residue.bond_angle[rt][j],
                             Residue.torsion[rt][j])
                res._atom[j].xyz = np.round(newH.xyz, 3)
                res._atom[j]._name = inv_map[j][1:]
                res._atom[j]._type = 25
            prev_res = res

    def writePdb(self, filename, start=None, end=None, l_start=None,
                 l_end=None, tp=0):
        """
        Write a PDB file in the same layout as Structure::writePdb.
        Atom names come from the residue template, hydrogens and unset atoms
        are skipped, and residues are numbered sequentially from 1. 
        A residue whose CA is still at the origin is written as a single 
        H placeholder line, which is used to mark a missing residue in a loop.
        tp=0 writes the whole structure, tp=1 writes backbone atoms only.  
        When l_start and l_end are given, residues inside that range are backbone-only, 
        which is used when side chains were not sampled for the regrown loop
        """
        # set start and end if not specified
        if start is None or start < 1:
            start = 1
        if end is None or end > self.numRes:
            end = self.numRes
        lines = []
        atom_nr = 1
        # write line in PDB format, update atom_nr
        def pdbline(name, resname, resnum, xyz, atom_nr):
            label = name if len(name) >= 4 else " " + name
            line_out = ("ATOM  {:>5} {:<4} {:>3}  {:>4}    "
                        "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
                        atom_nr, label, resname, resnum,
                        xyz[0], xyz[1], xyz[2], 1.0, 1.0))
            return atom_nr, line_out
        # go over selected range
        for i in range(start, end + 1):
            res = self._res[i]
            rtype = res._type
            resname = Residue.Name3[rtype]
            # if residue has zero for coordinates, save and continue
            if res._atom[ATM_CA].is_origin():
                atom_nr, lineout = pdbline("H", resname, i, res._atom[ATM_CA].xyz, atom_nr)
                lines.append(lineout)
                continue
            # check if to save only backbone or not
            backbone_only = bool(tp)
            if l_start is not None and l_end is not None:
                backbone_only = l_start <= i <= l_end
            # save each atom if not undefined or 
            for j, at in enumerate(res._atom):
                if at._type == UNDEF or at._type >= H_ATOM_TYPE:
                    continue
                if backbone_only and j >= NUM_BB_ATOM:
                    break
                name = str(Residue.cType[rtype][j])
                # skip virtual CB added for GLY
                if rtype == GLY and name == "CB":
                    continue
                atom_nr, lineout = pdbline(name, resname, i, at.xyz, atom_nr)
                lines.append(lineout)
        # save result in specified location
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(filename, "w") as f:
            f.writelines(lines)

# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

def MeanSQ(A, B, mode=1):
    """
    Squared deviation between two residues based on util.cpp:MeanSQ
    mode = 1: the four main-chain atoms (N, CA, C, O) 
    mode = 2: adds CB: (N, CA, C, O, CB)
    mode = 3: adds every non-H atom
    """
    if A._type != B._type:
        raise ValueError(f"RMSD between different residues: "
                         f"{A.name}@{A._posn} vs {B.name}@{B._posn}")
    sq = 0.0
    size = 0
    for k in range(4):
        sq += float(np.sum((A._atom[k].xyz - B._atom[k].xyz) ** 2))
        size += 1
    if mode == 2 and A._type != GLY:
        sq += float(np.sum((A._atom[5].xyz - B._atom[5].xyz) ** 2))
        size += 1
    elif mode == 3:
        for k in range(5, len(A._atom)):
            if (A._atom[k]._type < 21 and B._atom[k]._type < 21
                    and A._atom[k]._type != UNDEF and B._atom[k]._type != UNDEF):
                sq += float(np.sum((A._atom[k].xyz - B._atom[k].xyz) ** 2))
                size += 1
    return sq, size


def Root_MSD(A, B, Start1, End1, Start2, mode=1, normal=0):
    """
    RMSD between two structures without superposition. adaptation of util.cpp:Root_MSD.
    Residue Start1 to End1 of A is compared with Start2 to End2 of B.
    normal=1 applies the chain-length normalisation used for loop scoring.
    """
    sq_sum = 0.0
    size = 0
    for i in range(Start1, End1 + 1):
        j = Start2 + i - Start1
        s, n = MeanSQ(A._res[i], B._res[j], mode)
        sq_sum += s
        size += n
    rms = math.sqrt(sq_sum / size)
    if normal == 1:
        n_res = End1 - Start1 + 1
        rms = rms / (1 + math.log(math.sqrt(n_res / 6.0)))
    return rms

# ---------------------------------------------------------------------------
# Some helper functions for Structure parsing
# ---------------------------------------------------------------------------

def resnum_to_index(conf, resnum, chain=None):
    """
    Translate a PDB residue number into the position index.
    """
    # find the numRes that matches to pdbIndex resnum
    hits = [i for i in range(1, conf.numRes + 1)
            if conf._res[i]._pdbIndex == resnum
            and (chain is None or conf._res[i]._chainName == chain)]
    # raise error if res does not exist
    if not hits:
        where = "" if chain is None else f" of chain {chain}"
        raise ValueError(f"residue {resnum}{where} is not present in "
                         f"{conf._ProtName or 'the input structure'}")
    # raise warning for multiple instances of same pdbIndex resnum
    if len(hits) > 1:
        chains = ", ".join(conf._res[i]._chainName for i in hits)
        raise ValueError(f"residue number {resnum} occurs in more than one "
                         f"chain ({chains}); select one with --chain")
    return hits[0]

def blank_loop(pdb_lines, start_resnum, end_resnum, loopseq, chain=None):
    """
    Rewrite input pdb lines to insert blank atoms with the specified loopseq 
    sequence between selected anchors, telling DisGro which residues to build.
    Returns the new list of lines.
    """
    from .constants import _PLACEHOLDER
    loopseq = "".join(loopseq.split()).upper()
    atoms = [l for l in pdb_lines if l[:6] == "ATOM  "]
    # check if sequence is specified
    if not loopseq:
        raise ValueError("--loopseq is empty")
    # check if sequence contains any unknown residues
    unknown = sorted({c for c in loopseq if Residue.AAMap.get(c, "") not in Residue.AIMap})
    if unknown:
        raise ValueError(f"--loopseq contains unknown residue letters: {''.join(unknown)}")
    # Work out which chain the loop should be in
    if chain is None:
        chains = sorted({l[21] for l in atoms if int(l[22:26]) == start_resnum})
        if not chains:
            raise ValueError(f"residue {start_resnum} is not present in the input PDB")
        if len(chains) > 1:
            raise ValueError(f"residue {start_resnum} occurs in multiple chains. select one with --chain")
        chain = chains[0]
    # check if specified chain contains the start and end anchors
    for anchor in (start_resnum, end_resnum):
        if not any(l[21] == chain and int(l[22:26]) == anchor for l in atoms):
            raise ValueError(f"anchor residue {anchor} is not present in chain {chain} of the input PDB")
    # check if the loop size matches the loopseq
    loopsize = end_resnum - start_resnum - 1
    if len(loopseq) > loopsize:
        raise ValueError(f"--loopseq size does not match the range between {start_resnum} and {end_resnum}")
    # generate placeholder lines
    placeholders = []
    for k, letter in enumerate(loopseq):
        line_out = _PLACEHOLDER.format(serial=0, resname=Residue.AAMap[letter],
                                        chain=chain, resnum=start_resnum + 1 + k)
        placeholders.append(line_out)
    # construct new input pdb with placeholder atoms
    out = []
    inserted = False
    for line in pdb_lines:
        record = line[:6]
        if record in ("ATOM  ", "ANISOU") and line[21] == chain:
            nr = int(line[22:26])
            # if selected res already exist, skip to replace
            if start_resnum < nr < end_resnum:
                continue
            # insert placeholder lines
            if not inserted and record == "ATOM  " and nr >= end_resnum:
                out.extend(placeholders)      
                inserted = True
        out.append(line)
    # if placeholder lines have not been inserted, insert at end
    if not inserted:
        out.extend(placeholders)
    return renumber_pdb(out)

def renumber_pdb(lines):
    """
    Give the pdb lines consecutive atom numbers
    """
    out = []
    serial = 0
    for line in lines:
        record = line[:6]
        if record in ("ATOM  ", "HETATM"):
            serial += 1
            out.append(f"{record}{serial:5d}{line[11:]}")
        elif record == "ANISOU":
            out.append(f"{record}{serial:5d}{line[11:]}")
        else:
            out.append(line)
    return out