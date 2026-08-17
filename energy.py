# LOODIS energy terms, evaluated on the flatStructure version of the protein structues.

import numpy as np

from .constants import *
from .geom import Atom
from .potential import PF
from .residue import Residue

# ---------------------------------------------------------------------------
# helper functions for calculating LOODIS energy terms
# ---------------------------------------------------------------------------

def _lookup(t1, t2, d2):
    """
    LOODIS scores for atom-type pairs t1/t2 
    at squared distances d2
    """
    bins = (np.sqrt(d2) / H_INLO).astype(np.int64)
    np.clip(bins, 0, LOODIS_DIS_BIN - 1, out=bins)
    return PF.LOODIS[t1 - 1, t2 - 1, bins]

def _sqdist(A, B):
    """
    fast square distance used in screening
    """
    diff = A[:, None, :] - B[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def _select_atoms(flatstruc, res_mask, limits):
    """
    Indices of usable heavy atoms in the selected residues, capped by limits.
    """
    ok = flatstruc.heavy_mask()
    ok &= res_mask[flatstruc.ridx]
    ok &= flatstruc.slot < limits[flatstruc.ridx]
    return np.nonzero(ok)[0]


def _backbone_limits(flatstruc, Start, End, sidechains):
    """
    Per-residue atom-count caps used by the LOODIS sums.
    With side chains present every atom counts.  During backbone growth the
    C++ restricts residues inside the fragment to their first six slots (five
    for glycine); residues outside keep all their atoms.
    """
    limits = flatstruc.res_natom.astype(np.int64).copy()
    if not sidechains:
        six = np.where(flatstruc.res_type == GLY, 5, 6)
        idx = np.arange(len(limits))
        inside = (idx >= Start) & (idx <= End)
        limits = np.where(inside, np.minimum(limits, six), limits)
    return limits


# ---------------------------------------------------------------------------
# whole-structure energy
# ---------------------------------------------------------------------------

def loodis_e(flatstruc, Start, End, sidechains=True):
    """
    Total LOODIS energy of the loop against the whole protein.
    Port of cal_energy.cpp:1873.  Sums over ordered residue pairs
    between two selections: those in the loop between Start and End, 
    and the entire structure. skipping the two sequence neighbours, 
    and when both residues are inside the loop, counting each pair only once.
    """
    nres = flatstruc.numRes
    idx = np.arange(1, nres + 1)
    loop = np.arange(Start, End + 1)

    l_col = loop[:, None]
    i_row = idx[None, :]
    pair = ~((i_row < l_col + 2) & (i_row > l_col - 2)) # drop l-1, l, l+1
    inside = (i_row >= Start) & (i_row <= End)
    pair &= ~(inside & (i_row <= l_col - 2)) # count loop pairs once
    d_cent = np.linalg.norm(flatstruc.res_center[loop][:, None, :]
                            - flatstruc.res_center[idx][None, :, :], axis=-1)
    pair &= d_cent < CC_DIS_CUT
    limits = _backbone_limits(flatstruc, Start, End, sidechains)
    usable = flatstruc.heavy_mask() & (flatstruc.slot < limits[flatstruc.ridx])

    energy = 0.0
    # go over each res in selected loop
    for row, l in enumerate(loop):
        # skip res if no partners
        partners = idx[pair[row]]
        if partners.size == 0:
            continue
        # skip res if no usable atoms
        a_idx = np.nonzero(usable & (flatstruc.ridx == l))[0]
        if a_idx.size == 0:
            continue
        # skip if no usable partners
        pmask = np.zeros(nres + 2, dtype=bool)
        pmask[partners] = True
        b_idx = np.nonzero(usable & pmask[flatstruc.ridx])[0]
        if b_idx.size == 0:
            continue
        # calculate square distances, skip if not within squaredist cutoff
        d2 = _sqdist(flatstruc.xyz[a_idx], flatstruc.xyz[b_idx])
        close = d2 < LOODIS_CUT_SQ
        if not close.any():
            continue
        # calc energy from lookup table for distance and atom types
        ta = np.broadcast_to(flatstruc.atype[a_idx][:, None], d2.shape)[close]
        tb = np.broadcast_to(flatstruc.atype[b_idx][None, :], d2.shape)[close]
        energy += float(_lookup(ta, tb, d2[close]).sum())
    return energy


def calE(flatstruc, Start, End, sidechains=True):
    """
    Total energy of a conformation port of calE() in cal_energy.cpp.
    Returns the total energy and enArr.  With only EM_LOODIS this is 
    just the LOODIS term.
    """
    enArr = np.zeros(ENERGY_TYPES)
    if PF.cal[EM_LOODIS]:
        enArr[E_LOODIS] = loodis_e(flatstruc, Start, End, sidechains)
    return float(enArr.sum()), enArr


# ---------------------------------------------------------------------------
# per-residue energy, batched over candidate states
# ---------------------------------------------------------------------------

def one_res_en(flatstruc, cand_xyz, cand_types, cand_ref, position, start, end,
               Start, End, etype):
    """
    Energy of a candidate residue against residues in loop from start to end
    cand_xyz : (S, 6, 3) Backbone array of each candidate state.
    During growth these hold C, O, CB, of pos, N, CA of pos+1
    cand_types : (6,) Atom type of each slot.
    cand_ref : (S, 3) Residue centre for etype 0 or backbone centre, 
    for each candidate, used for the proximity prefilter.
    etype: 0 = loop-closure scoring, 1 = chain before the fragment, 
    2 = chain after fragment.
    """
    S = cand_xyz.shape[0]
    energy = np.zeros(S)
    end = min(end, flatstruc.numRes)
    if start > end:
        return energy

    part = np.arange(start, end + 1)
    part = part[~np.all(flatstruc.res_center[part] == 0.0, axis=1)]
    if part.size == 0:
        return energy

    # Proximity prefilter. the reference point differs between the two modes.
    if etype == 0:
        thresh = (Residue.size[flatstruc.res_type[position]] + flatstruc.res_size[part]
                  + CUB_SIZE)
    else:
        thresh = Residue.bb_size + flatstruc.res_size[part] + CUB_SIZE
    d_cent = np.linalg.norm(cand_ref[:, None, :] - flatstruc.res_center[part][None, :, :],
                            axis=-1)
    near = d_cent < thresh # (S, n_part)
    keep = near.any(axis=0)
    if not keep.any():
        return energy
    part, near = part[keep], near[:, keep]

    # Residues inside the fragment contribute their backbone slots only.
    limits = flatstruc.res_natom.astype(np.int64).copy()
    inside = (part >= Start) & (part <= End)
    limits[part[inside]] = NUM_BB_ATOM
    pmask = np.zeros(flatstruc.numRes + 2, dtype=bool)
    pmask[part] = True
    b_idx = _select_atoms(flatstruc, pmask, limits)
    if b_idx.size == 0:
        return energy

    b_xyz = flatstruc.xyz[b_idx]
    b_type = flatstruc.atype[b_idx]
    b_res = flatstruc.ridx[b_idx]
    b_slot = flatstruc.slot[b_idx]

    col = np.full(flatstruc.numRes + 2, -1, dtype=np.int64)
    col[part] = np.arange(part.size)
    near_atom = near[:, col[b_res]] # (S, n_b)

    j_ok = np.nonzero((cand_types != UNDEF) & (cand_types < H_ATOM_TYPE))[0]
    if j_ok.size == 0:
        return energy
    allowed = _adjacency_mask(j_ok, b_res, b_slot, position, end, Start, etype)

    cand = cand_xyz[:, j_ok, :] # (S, J, 3)
    diff = cand[:, :, None, :] - b_xyz[None, None, :, :]
    d2 = np.einsum("sjbk,sjbk->sjb", diff, diff)

    ok = d2 <= PF_DIS_CUT_SQUARE
    ok &= allowed[None, :, :]
    ok &= near_atom[:, None, :]
    ok &= ~np.all(cand == 0.0, axis=-1)[:, :, None] # unplaced atom
    if not ok.any():
        return energy

    s_i, j_i, b_i = np.nonzero(ok)
    vals = _lookup(cand_types[j_ok][j_i], b_type[b_i], d2[s_i, j_i, b_i])
    return np.bincount(s_i, weights=vals, minlength=S)


def _adjacency_mask(j_ok, b_res, b_slot, position, end, Start, etype):
    """
    Allowed-pair mask of shape ``(len(j_ok), n_partner_atoms)``.
    Reproduces the two exclusions ``one_res_en`` applies when scoring against
    the chain *before* the fragment (``etype == 1``): the atoms of the directly
    preceding residue that are separated from the new C/CB by three bonds or
    fewer, and the C/O pair against the last partner residue once growth has
    moved past the fragment start.
    """
    allowed = np.ones((j_ok.size, b_res.size), dtype=bool)
    if etype != 1:
        return allowed

    j = j_ok[:, None]
    k = b_slot[None, :]

    prev = (b_res == position - 1)[None, :]
    allowed &= ~(prev & (k == ATM_C) & ((j == ATM_C) | (j == ATM_CB)))

    if position > Start:
        last = (b_res == end)[None, :]
        jco = (j == ATM_C) | (j == ATM_O)
        kco = (k == ATM_C) | (k == ATM_O)
        allowed &= ~(last & jco & kco)
    return allowed


def one_res_en_sc(body, cand_xyz, cand_types, cand_scc, position, Start, End,
                  sc_size, require_named=False):
    """
    Energy of a candidate side chain against the rest of the protein.
    Port of cal_energy.cpp:398.  Only side-chain atoms of the candidate
    contribute. the three partner ranges are the chain before the residue, 
    the chain after the fragment, and the backbone of the remaining loop.

    require_named reproduces a quirk of the C++: partner atoms whose 
    _name is empty are skipped.  Atoms that were absent from the input PDB 
    never get a name, and growth copies coordinates only, 
    so the freshly grown backbone of the loop is invisible to this term
    in the reference implementation.  Pass False for the more defensible
    behaviour of counting every placed heavy atom.
    """
    S = cand_xyz.shape[0]
    energy = np.zeros(S)
    nres = body.numRes

    j_ok = np.nonzero((cand_types != UNDEF) & (cand_types < H_ATOM_TYPE))[0]
    j_ok = j_ok[j_ok >= NUM_BB_ATOM]
    if j_ok.size == 0:
        return energy
    cand = cand_xyz[:, j_ok, :]

    def accumulate(part, limits, force_include=None):
        nonlocal energy
        if part.size == 0:
            return
        thresh = sc_size + body.res_size[part] + CUB_SIZE
        d = np.linalg.norm(cand_scc[:, None, :] - body.res_center[part][None, :, :],
                           axis=-1)
        near = d < thresh
        if force_include is not None:
            near |= force_include[None, :]
        keep = near.any(axis=0)
        if not keep.any():
            return
        sel, near, lim_sel = part[keep], near[:, keep], limits[keep]

        pmask = np.zeros(nres + 2, dtype=bool)
        pmask[sel] = True
        lim = body.res_natom.astype(np.int64).copy()
        lim[sel] = lim_sel
        b_idx = _select_atoms(body, pmask, lim)
        b_idx = b_idx[body.slot[b_idx] != 4]     # backbone H, skipped by the C++
        if require_named:
            b_idx = b_idx[body.named[b_idx]]
        if b_idx.size == 0:
            return

        col = np.full(nres + 2, -1, dtype=np.int64)
        col[sel] = np.arange(sel.size)
        near_atom = near[:, col[body.ridx[b_idx]]]
        b_xyz = body.xyz[b_idx]
        b_type = body.atype[b_idx]

        diff = cand[:, :, None, :] - b_xyz[None, None, :, :]
        d2 = np.einsum("sjbk,sjbk->sjb", diff, diff)
        ok = (d2 <= PF_DIS_CUT_SQUARE) & near_atom[:, None, :]
        if not ok.any():
            return
        s_i, j_i, b_i = np.nonzero(ok)
        vals = _lookup(cand_types[j_ok][j_i], b_type[b_i], d2[s_i, j_i, b_i])
        energy = energy + np.bincount(s_i, weights=vals, minlength=S)

    part = np.arange(1, position)
    if part.size:
        accumulate(part, body.res_natom[part].astype(np.int64),
                   force_include=(part >= Start))
    part = np.arange(End + 1, nres + 1)
    if part.size:
        accumulate(part, body.res_natom[part].astype(np.int64))
    part = np.arange(position + 1, End + 1)
    if part.size:
        accumulate(part, np.full(part.size, NUM_BB_ATOM, dtype=np.int64))

    return energy

# ---------------------------------------------------------------------------
# clash detection
# ---------------------------------------------------------------------------

def BBClash_detection(body, Start, End):
    """
    Count severe backbone clashes per loop residue cal_energy.cpp.
    Returns (res_indices, clash_counts) for residues with a non-zero count.
    A pair closer than 0.5 of the summed vdW radii is weighted five times
    Glycine and alanine loop residues are skipped.
    """
    nres = body.numRes
    heavy = body.heavy_mask()
    res_idx, clash_num = [], []
    all_limits = body.res_natom.astype(np.int64)

    # go over each residue in selected range
    for l in range(Start, End + 1):
        if body.res_type[l] in (ALA, GLY):
            continue
        # create selection for rest of protein
        part = np.arange(1, nres + 1)
        part = part[(part < l) | (part > End)]
        if part.size == 0:
            continue

        # calculate tresholds and distances for atoms in residue vs protein
        thresh_c = body.res_size[l] + body.res_size[part] + CUB_SIZE
        thresh_b = Residue.bb_size + body.res_size[part] + CUB_SIZE
        d_c = np.linalg.norm(body.res_center[l] - body.res_center[part], axis=-1)
        d_b = np.linalg.norm(body.res_bbc[l] - body.res_center[part], axis=-1)
        part = part[(d_c < thresh_c) | (d_b < thresh_b)]
        if part.size == 0:
            continue

        lo = int(body.res_start[l])
        a_idx = lo + np.arange(NUM_BB_ATOM)
        a_idx = a_idx[heavy[a_idx]]
        if a_idx.size == 0:
            continue

        pmask = np.zeros(nres + 2, dtype=bool)
        pmask[part] = True
        b_idx = _select_atoms(body, pmask, all_limits)
        if b_idx.size == 0:
            continue

        d2 = _sqdist(body.xyz[a_idx], body.xyz[b_idx])
        ok = d2 <= PF_DIS_CUT_SQUARE
        ok &= _bb_bonded_mask(body.slot[a_idx], body.ridx[b_idx],
                              body.slot[b_idx], l)
        if not ok.any():
            continue

        a_i, b_i = np.nonzero(ok)
        r = (Atom.radius[body.atype[a_idx][a_i]]
             + Atom.radius[body.atype[b_idx][b_i]])
        quot = np.sqrt(d2[a_i, b_i]) / r
        clash = quot <= VDW_CLASH_CUTOFF
        count = int(np.sum(clash & (quot >= 0.5)) + 5 * np.sum(clash & (quot < 0.5)))
        if count:
            res_idx.append(l)
            clash_num.append(count)
    return res_idx, clash_num


def _bb_bonded_mask(a_slot, b_res, b_slot, l):
    """
    Exclude nearby bonded atoms from neighboring residues from clash evaluation
    by creating a mask excluding through-bond neighbours
    """
    allowed = np.ones((a_slot.size, b_res.size), dtype=bool)
    j = a_slot[:, None]
    k = b_slot[None, :]

    nxt = (b_res == l + 1)[None, :]
    allowed &= ~(nxt & (j == ATM_CA) & (k == ATM_CA))
    allowed &= ~(nxt & (j == ATM_C) & ((k == ATM_N) | (k == ATM_CA)))
    allowed &= ~(nxt & (j == ATM_O) & (k == ATM_N))

    prv = (b_res == l - 1)[None, :]
    allowed &= ~(prv & (k == ATM_CA) & (j == ATM_CA))
    allowed &= ~(prv & (k == ATM_C) & ((j == ATM_N) | (j == ATM_CA)))
    allowed &= ~(prv & (k == ATM_O) & (j == ATM_N))
    return allowed
