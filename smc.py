# implementation of the sequential Monte Carlo used to build the loop

import math
import time
import numpy as np

from .constants import *
from .closure import analytic_closure
from .energy import BBClash_detection, calE, one_res_en, one_res_en_sc
from .geom import (SampleOne, angle_batch, box_muller_batch, calCo_batch,
                   frand, rng, torsion_batch)
from .potential import PF
from .residue import Residue
from .sampling import (EmpiricalDistances, load_joint_angles, sample_distance,
                       sample_sc_angles)

def ExpandNumStates(num_states, frag_length, kind):
    """
    Broadcast a single state count across a Structure fragment
    """
    if len(num_states) == 1:
        return [num_states[0]] * frag_length
    if len(num_states) == frag_length:
        return list(num_states)
    raise ValueError(f"fragment length {frag_length} does not match the "
                     f"{len(num_states)} inputted numbers of {kind} states")


class SMC:
    """
    Loop sampler for one protein and one loop range.
    start and end are residue indices applied to the Structure object conf 
    (1-based, matching the PDB numbering the reference program uses).  
    The residues strictly between them are rebuilt; start and end act as anchors.
    """

    def __init__(self, conf, start, end, num_conf=5000,
                 num_distance_states=(32,), num_angle_states=(0,),
                 confkeep=1, sample_sc=False, num_sc_states=5, ang_type=2,
                 evaluate=False, close=True, no_score=False, temperature=1.0,
                 sc_accumulate_energy=False, verbose=True):
        self.Conf = conf
        self.Start = start
        self.End = end
        self.NumConf = num_conf
        self.confkeep = confkeep
        self.sample_sc = sample_sc
        self.numSCStates = num_sc_states
        self.AngType = ang_type
        self.Eval = evaluate
        self.Close = close
        self.noScore = no_score
        self.T = temperature
        self.verbose = verbose
        
        # original code accumulates energy per independent sampling run. 
        # this is probably a bug, so False by default.
        self.sc_accumulate_energy = sc_accumulate_energy

        frag_length = end - start
        self.NumAngleStates = ExpandNumStates(list(num_angle_states),
                                              frag_length, "angle")
        self.NumDistanceStates = ExpandNumStates(list(num_distance_states),
                                                 frag_length, "distance")

        self.dist = EmpiricalDistances.load_default()
        self.joint_angle = load_joint_angles()

        self.base = conf.flatten()
        self.numRes = self.base.numRes
        self.loop_lo = int(self.base.res_start[start])
        self.loop_hi = int(self.base.res_start[end + 1])

        self.LoopStore = [] # coordinates of accepted loops
        self.LoopEnergy = [] # energy of accepted loops
        self.minEnergy = 1e4
        self.NumClosedconf = 0

    # setup
    def PreProcess(self):
        """
        Switch distance states to angle states for a terminal fragment.
        Used as setup to prepare for growing a loop
        """
        if self.End == self.numRes:
            if self.verbose:
                print("Editing end fragment!")
            self.NumAngleStates = [a + d for a, d in
                                   zip(self.NumAngleStates, self.NumDistanceStates)]
            self.NumDistanceStates = [0] * len(self.NumDistanceStates)
            self.Close = False

    # one growth step
    def grow_one(self, work, position, tmpEnd, endpt, state):
        """
        Grow residue at position by one step.
        Returns False when the remaining end-to-end distance falls outside the
        range the empirical tables cover, which means this trial cannot reach
        the anchor and is abandoned.
        """
        rel = position - self.Start
        rem = tmpEnd - position
        n_ang = self.NumAngleStates[rel]
        n_dis = self.NumDistanceStates[rel]
        n_states = n_ang + n_dis
        if n_states == 0:
            raise ValueError("no sampling states requested for this residue")
        res_type = int(work.res_type[position])

        bb = np.zeros((n_states, 3))
        if n_dis > 0:
            ok = self._propose_from_distance(work, position, rem, endpt,
                                             res_type, n_states, bb)
            if not ok:
                return False
        else:
            # Pure angle sampling: only reachable for a terminal fragment,
            # where PreProcess has converted every distance state.
            bb[:, 0] = frand(-PI, PI, n_states)
            bb[:, 1] = frand(-PI, PI, n_states)
            bb[:, 2] = PI

        cand, ctypes = self._build_candidates(work, position, bb)
        cand_bbc = _backbone_center(cand, ctypes)

        energy = one_res_en(work, cand, ctypes, cand_bbc, position,
                            1, position - 1, self.Start, self.End, 1)
        if self.End != self.numRes:
            energy = energy + one_res_en(work, cand, ctypes, cand_bbc, position,
                                         self.End + 1, self.numRes,
                                         self.Start, self.End, 2)
        energy = energy + self._anchor_terms(work, position, cand, ctypes)

        minE = energy.min()
        prob = np.power(EXPO, (minE - energy) * 0.5)
        if not np.isfinite(prob).all():
            raise FloatingPointError(f"grow_one probability overflow at "
                                     f"residue {position}: {energy}")
        if prob.sum() == 0:
            prob = np.ones(n_states)
        chosen = SampleOne(prob)

        state.weight += math.log(prob.sum() / prob[chosen])
        state.energy += float(energy[chosen])

        # Write the accepted geometry back: C/O/CB belong to `position`, the
        # N/CA slots of the candidate hold the next residue's leading atoms.
        lo = int(work.res_start[position])
        nxt = int(work.res_start[position + 1])
        work.xyz[lo + ATM_C] = cand[chosen, ATM_C]
        work.xyz[lo + ATM_O] = cand[chosen, ATM_O]
        work.xyz[nxt + ATM_N] = cand[chosen, ATM_N]
        work.xyz[nxt + ATM_CA] = cand[chosen, ATM_CA]
        if work.res_type[position] != GLY:
            work.xyz[lo + ATM_CB] = cand[chosen, ATM_CB]
        work.update_center(position, sidechain=False)
        return True

    def _propose_from_distance(self, work, position, rem, endpt, res_type,
                               n_states, bb):
        """
        Fill backbone with phi/psi/omega drawn via the distance tables.
        """
        L = LARGE_NUM_DISTANCE_STATES
        dist = self.dist
        n_atom = work.atom(position, ATM_N)
        ca_atom = work.atom(position, ATM_CA)

        ee_c = float(np.linalg.norm(ca_atom - endpt))
        lo_c, hi_c = dist.range(1, rem)
        if not (lo_c <= ee_c <= hi_c):
            return False

        sam_c = sample_distance(dist, n_atom, ca_atom, endpt,
                                Residue.bond_angle[res_type][ATM_C],
                                Residue.bond_length[res_type][ATM_C],
                                ee_c, 1, rem, L)
        alive = np.isfinite(sam_c).all(axis=1)

        ee_n = np.linalg.norm(sam_c - endpt, axis=1)
        lo_n, hi_n = dist.range(0, rem - 1)
        alive &= np.isfinite(ee_n) & (ee_n >= lo_n) & (ee_n <= hi_n)
        if not alive.any():
            return False

        safe_c = np.where(alive[:, None], sam_c, ca_atom + 1.0)
        safe_ee = np.where(alive, ee_n, 0.5 * (lo_n + hi_n))
        with np.errstate(invalid="ignore", divide="ignore"):
            sam_n = sample_distance(dist, ca_atom, safe_c, endpt,
                                    Residue.bond_angle[res_type][ATM_N],
                                    Residue.bond_length[res_type][ATM_N],
                                    safe_ee, 0, rem - 1, L)
        alive &= np.isfinite(sam_n).all(axis=1)
        if not alive.any():
            return False

        prevC = work.atom(position - 1, ATM_C)
        with np.errstate(invalid="ignore"):
            phi = np.radians(torsion_batch(prevC, n_atom, ca_atom, sam_c))
            psi = np.radians(torsion_batch(n_atom, ca_atom, sam_c, sam_n))
        omega = box_muller_batch(PI, 4, L)
        alive &= np.isfinite(phi) & np.isfinite(psi)
        if not alive.any():
            return False

        # Bin the proposals and weight them by the observed joint phi/psi
        # frequency for this residue type.
        binNum = 180 // BBTbinSize
        phi_deg = np.where(alive, np.degrees(phi), 0.0)
        psi_deg = np.where(alive, np.degrees(psi), 0.0)
        phi_bin = np.clip(np.floor(phi_deg / BBTbinSize).astype(np.int64)
                          + binNum, 0, TORBIN - 1)
        psi_bin = np.clip(np.floor(psi_deg / BBTbinSize).astype(np.int64)
                          + binNum, 0, TORBIN - 1)
        counts = np.where(alive, self.joint_angle[res_type][phi_bin, psi_bin], 0)
        total = int(counts.sum())

        if total == 0:
            # No proposal landed in a populated bin; fall back to the live
            # states with equal weight (smc.cpp:622-632).
            pick = rng().choice(np.nonzero(alive)[0], size=n_states)
        else:
            cum = np.cumsum(counts)
            draws = rng().integers(0, total, n_states)
            pick = np.searchsorted(cum, draws, side="right")

        lo_edge = (phi_bin[pick] - binNum) * PI / binNum
        bb[:, 0] = frand(lo_edge, lo_edge + PI / binNum)
        lo_edge = (psi_bin[pick] - binNum) * PI / binNum
        bb[:, 1] = frand(lo_edge, lo_edge + PI / binNum)
        bb[:, 2] = omega[pick]
        return True

    def _build_candidates(self, work, position, bb):
        """
        Build candidate backbones for a given position
        Batched calBBCo: backbone slots for every candidate state.
        """
        cur = int(work.res_start[position])
        prev = int(work.res_start[position - 1])
        tp = int(work.res_type[position])
        tn = int(work.res_type[position + 1])
        prevC = work.xyz[prev + ATM_C]
        curN = work.xyz[cur + ATM_N]
        curCA = work.xyz[cur + ATM_CA]
        bl, ba = Residue.bond_length, Residue.bond_angle
        phi, psi, omega = bb[:, 0], bb[:, 1], bb[:, 2]
        S = bb.shape[0]

        C = calCo_batch(prevC, curN, curCA, bl[tp][ATM_C], ba[tp][ATM_C], phi)
        O = calCo_batch(curN, curCA, C, bl[tp][ATM_O], ba[tp][ATM_O], psi + PI)
        N = calCo_batch(curN, curCA, C, bl[tn][ATM_N], ba[tn][ATM_N], psi)
        CA = calCo_batch(curCA, C, N, bl[tn][ATM_CA], ba[tn][ATM_CA], omega)

        cand = np.zeros((S, NUM_BB_ATOM, 3))
        ctypes = np.full(NUM_BB_ATOM, UNDEF, dtype=np.int64)
        n_slot = min(NUM_BB_ATOM, int(work.res_natom[position]))
        ctypes[:n_slot] = work.atype[cur:cur + n_slot]
        cand[:, ATM_H] = work.xyz[cur + ATM_H]
        cand[:, ATM_N] = N
        cand[:, ATM_CA] = CA
        cand[:, ATM_C] = C
        cand[:, ATM_O] = O
        if tp != GLY:
            cand[:, ATM_CB] = calCo_batch(curN, C, curCA, bl[tp][ATM_CB],
                                          ba[tp][ATM_CB], PI * 122.55 / 180)
        return cand, ctypes

    def _anchor_terms(self, work, position, cand, ctypes):
        """
        Extra LOODIS terms the C++ adds by hand in grow_one.
        The growing residue is scored against the CA and C of the anchor
        residue End, and the newly placed CA against the N of position.
        Neither pair is covered by one_res_en, which stops at End - 1.
        """
        S = cand.shape[0]
        out = np.zeros(S)
        End = self.End
        end_lo = int(work.res_start[End])
        anchors = work.xyz[end_lo + ATM_CA:end_lo + ATM_C + 1]
        anchor_types = work.atype[end_lo + ATM_CA:end_lo + ATM_C + 1]

        keep = (ctypes != UNDEF) & (ctypes < H_ATOM_TYPE)
        sub = cand[:, keep, :]
        placed = ~np.all(sub == 0.0, axis=-1)
        d2 = np.sum((sub[:, :, None, :] - anchors[None, None, :, :]) ** 2, axis=-1)
        ok = (d2 <= PF_DIS_CUT_SQUARE) & placed[:, :, None]
        if ok.any():
            s_i, j_i, a_i = np.nonzero(ok)
            bins = np.clip((np.sqrt(d2[s_i, j_i, a_i]) / H_INLO).astype(np.int64),
                           0, LOODIS_DIS_BIN - 1)
            vals = PF.LOODIS[ctypes[keep][j_i] - 1, anchor_types[a_i] - 1, bins]
            out += np.bincount(s_i, weights=vals, minlength=S)

        # CA of position+1 against N of position.
        cur = int(work.res_start[position])
        nN = work.xyz[cur + ATM_N]
        d2 = np.sum((cand[:, ATM_CA, :] - nN) ** 2, axis=-1)
        close = d2 <= PF_DIS_CUT_SQUARE
        if close.any():
            bins = np.clip((np.sqrt(d2[close]) / H_INLO).astype(np.int64), 0,
                           LOODIS_DIS_BIN - 1)
            out[close] += PF.LOODIS[int(ctypes[ATM_CA]) - 1,
                                    int(work.atype[cur + ATM_N]) - 1, bins]
        return out

    def trial(self):
        """
        Grow and close one loop conformation.
        Returns the working flatStructures and a small state object, 
        or (None, state) if growth failed.
        """
        work = self.base.copy()
        state = _TrialState()
        Start, End = self.Start, self.End
        endpt = work.atom(End, ATM_C)

        grow_success = False
        grow_pre = False

        last = End if End == self.numRes else End - 2
        for i in range(Start, last):
            grow_success = self.grow_one(work, i, End, endpt, state)
            if not grow_success:
                return None, state

        if End - Start < 4 and End != self.numRes:
            grow_pre = self.grow_one(work, End - 2, End, endpt, state)
            if grow_pre:
                grow_success = self.grow_one(work, End - 1, End, endpt, state)
                # Place the final CA in line with the following atoms, which
                # grow_one does not guarantee: omega = pi.
                t_end = int(work.res_type[End])
                t_next = int(work.res_type[End + 1])
                work.xyz[work.index(End, ATM_CA)] = calCo_batch(
                    work.atom(End + 1, ATM_CA), work.atom(End + 1, ATM_N),
                    work.atom(End, ATM_C),
                    Residue.bond_length[t_end][ATM_C],
                    Residue.bond_angle[t_next][ATM_N], PI)
                if grow_success:
                    state.closed = _is_closed(work, End)

        if self.Close and not state.closed:
            analytic_closure(work, End - 2, Start, End)

        if self.Close and End != self.numRes:
            if grow_success and grow_pre:
                state.closed = _is_closed(work, End)
            elif grow_pre:
                state.closed = _is_closed(work, End - 1)
            else:
                state.closed = _is_closed(work, End - 2)
                for _ in range(MAX_CLOSURE_RETRIES):
                    if state.closed:
                        break
                    analytic_closure(work, End - 2, Start, End,
                                     jitter=CLOSURE_JITTER)
                    state.closed = _is_closed(work, End - 2)

        if End == self.numRes:
            _cap_c_terminus(work, End)

        return work, state

    def run(self):
        """
        Generate, rank and return loop conformations ported from SMC::Wholeproc.
        """
        self.PreProcess()
        if not self.noScore:
            energy, enArr = calE(self.base, self.Start, self.End, True)
            self.Conf._energy = energy
            self.Conf._enArr = enArr
            if self.verbose:
                print(f"Native energy over the loop range: {energy:.4f}")

        if self.verbose:
            print("DiSGro in progress ...")
        t0 = time.time()
        for _ in range(self.NumConf):
            work, state = self.trial()
            if work is None:
                continue
            terminal = self.End == self.numRes
            if not (state.closed or terminal):
                continue
            self.NumClosedconf += 1

            if not self.noScore and state.closed:
                work.update_centers(self.Start, self.End, sidechain=False)
                state.energy, _ = calE(work, self.Start, self.End, False)

            if (state.energy < self.minEnergy + ENERGY_CUTOFF
                    and len(self.LoopStore) < MAX_STORED_LOOPS):
                self.minEnergy = min(self.minEnergy, state.energy)
                if not self.noScore:
                    self.LoopStore.append(work.xyz[self.loop_lo:self.loop_hi].copy())
                    self.LoopEnergy.append(state.energy)
            if self.NumClosedconf >= MAX_CLOSED_CONF:
                break
        if self.verbose:
            print(f"Conformational sampling done in {time.time() - t0:.1f} s; "
                  f"{self.NumClosedconf} closed, {len(self.LoopStore)} stored")

        return self._rank()

    def _rank(self):
        """
        Keep the best conformations, reject clashes, optionally add side chains.
        """
        if not self.LoopStore:
            return []
        order = np.argsort(self.LoopEnergy)[:self.confkeep]
        results = []
        for idx in order:
            work = self.base.copy()
            work.xyz[self.loop_lo:self.loop_hi] = self.LoopStore[idx]
            work.update_centers(self.Start, self.End, sidechain=False)

            _, clashes = BBClash_detection(work, self.Start, self.End)
            if any(n > MAX_CLASH_PER_RESIDUE for n in clashes):
                continue

            energy = self.LoopEnergy[idx]
            if self.sample_sc:
                self.grow_sc(work)
                if self.Eval:
                    work.update_centers(self.Start, self.End, sidechain=True)
                    energy, _ = calE(work, self.Start, self.End, True)
            results.append(LoopResult(energy, work.xyz[self.loop_lo:self.loop_hi].copy()))

        results.sort(key=lambda r: r.energy)
        if self.verbose:
            print(f"Calculation done: {len(results)} conformations kept")
        return results

    def to_structure(self, result):
        """
        Rebuild a full Structure object with this loop grafted in.
        """
        flatstruc = self.base.copy()
        flatstruc.xyz[self.loop_lo:self.loop_hi] = result.xyz
        flatstruc.update_centers(self.Start, self.End, sidechain=self.sample_sc)
        out = self.Conf.copy()
        flatstruc.write_back(out)
        out._energy = result.energy
        out.fill_atom_names()
        return out

    def grow_sc(self, work):
        """
        Grow side chains over the loop, ported from Structure::grow_sc.
        Residues are visited in order and each is chosen by Boltzmann weight
        against the structure as it stands, so earlier choices constrain later
        ones.
        """
        Start, End = self.Start, self.End
        n_states = self.numSCStates
        to_sample = [i for i in range(Start, End + 1)
                     if work.res_type[i] not in (GLY, ALA)]
        carried = np.zeros(n_states)

        for n in to_sample:
            rtype = int(work.res_type[n])
            angles = sample_sc_angles(rtype, n_states, self.AngType)
            cand = _build_sidechains(work, n, angles)
            ctypes = work.atype[work.residue_slice(n)]
            scc = _sidechain_center(cand, ctypes)
            energy = one_res_en_sc(work, cand, ctypes, scc, n, Start, End,
                                   Residue.sc_size[rtype])
            if self.sc_accumulate_energy:
                energy = energy + carried
                carried = energy

            minE = energy.min()
            prob = np.power(EXPO, 0.5 * (minE - energy) / self.T)
            if not np.isfinite(prob).all():
                raise FloatingPointError(f"grow_sc probability overflow at {n}")
            if prob.sum() == 0:
                prob = np.ones(n_states)
            chosen = SampleOne(prob)

            sl = work.residue_slice(n)
            work.xyz[sl] = cand[chosen]
            work.res_scc[n] = scc[chosen]
            work.update_center(n, sidechain=True)

class _TrialState:
    __slots__ = ("energy", "weight", "closed")

    def __init__(self):
        self.energy = 0.0
        self.weight = 0.0
        self.closed = False

class LoopResult:
    """
    One accepted loop: its energy and the coordinates of the loop block.
    """
    __slots__ = ("energy", "xyz")

    def __init__(self, energy, xyz):
        self.energy = energy
        self.xyz = xyz

    def __repr__(self):
        return f"LoopResult(energy={self.energy:.3f})"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _backbone_center(cand, ctypes):
    """
    Backbone centre of each candidate.
    """
    ok = (ctypes != UNDEF) & (ctypes <= 20)
    ok = np.broadcast_to(ok, cand.shape[:2]) & ~np.all(cand == 0.0, axis=-1)
    n = ok.sum(axis=1, keepdims=True)
    total = (cand * ok[:, :, None]).sum(axis=1)
    return np.divide(total, n, out=np.zeros_like(total), where=n > 0)


def _sidechain_center(cand, ctypes):
    ok = (ctypes != UNDEF) & (ctypes < H_ATOM_TYPE)
    ok = np.array(ok, dtype=bool).copy()
    ok[:NUM_BB_ATOM] = False
    ok2 = np.broadcast_to(ok, cand.shape[:2])
    n = ok2.sum(axis=1, keepdims=True)
    total = (cand * ok2[:, :, None]).sum(axis=1)
    return np.divide(total, n, out=np.zeros_like(total), where=n > 0)


def _build_sidechains(work, res, angles):
    """
    Batched calSCCo for all atoms of res for every chi proposal.
    """
    rtype = int(work.res_type[res])
    sl = work.residue_slice(res)
    n_slot = int(work.res_natom[res])
    S = angles.shape[0]
    out = np.repeat(work.xyz[sl][None, ...], S, axis=0)

    rot = 0
    for j in range(NUM_BB_ATOM, n_slot):
        p = Residue.prev_atom[rtype][j]
        tor = Residue.torsion[rtype][j]
        if tor == -1234:
            torsion_angle = angles[:, rot]
            rot += 1
        else:
            torsion_angle = np.full(S, tor)
        out[:, j] = calCo_batch(out[:, p[0]], out[:, p[1]], out[:, p[2]],
                                Residue.bond_length[rtype][j],
                                Residue.bond_angle[rtype][j], torsion_angle)
    return out


def _is_closed(work, End):
    """
    apply same checks as Structure::IsClosed on the flat arrays.
    """
    n = work.atom(End, ATM_N)
    ca = work.atom(End, ATM_CA)
    c = work.atom(End, ATM_C)
    n2 = work.atom(End + 1, ATM_N)
    ca2 = work.atom(End + 1, ATM_CA)

    d1 = float(np.linalg.norm(n - ca))
    d2 = float(np.linalg.norm(ca - c))
    if not (CLOSED_N_CA_MIN < d1 < CLOSED_N_CA_MAX):
        return False
    if not (CLOSED_CA_C_MIN < d2 < CLOSED_CA_C_MAX):
        return False
    a1 = float(angle_batch(n, ca, c))
    if not (CLOSED_ANG_NCAC_MIN < a1 < CLOSED_ANG_NCAC_MAX):
        return False
    a2 = float(angle_batch(ca, c, n2))
    if not (CLOSED_ANG_CACN_MIN < a2 < CLOSED_ANG_CACN_MAX):
        return False
    t1 = float(torsion_batch(ca, c, n2, ca2))
    return abs(t1) > CLOSED_OMEGA_MIN


def _cap_c_terminus(work, End):
    """
    Place C, O, OXT and CB of a C-terminal residue
    """
    t_end = int(work.res_type[End])
    bl, ba = Residue.bond_length, Residue.bond_angle
    phi = frand(-PI, PI)
    psi = frand(-PI, PI)

    c = calCo_batch(work.atom(End - 1, ATM_C), work.atom(End, ATM_N),
                    work.atom(End, ATM_CA), bl[t_end][ATM_C], ba[t_end][ATM_C], phi)
    work.xyz[work.index(End, ATM_C)] = c
    work.xyz[work.index(End, ATM_O)] = calCo_batch(
        work.atom(End, ATM_N), work.atom(End, ATM_CA), c,
        bl[t_end][ATM_O], ba[t_end][ATM_O], psi)
    if t_end != GLY:
        work.xyz[work.index(End, ATM_CB)] = calCo_batch(
            work.atom(End, ATM_N), work.atom(End, ATM_C), work.atom(End, ATM_CA),
            bl[t_end][ATM_CB], ba[t_end][ATM_CB], PI * 122.55 / 180)
    work.update_center(End, sidechain=False)
