# all global variables are stored here

import math
import os
from importlib.resources import files, as_file

# ---------------------------------------------------------------------------
# data files
# ---------------------------------------------------------------------------

# Directory holding the parameter files shipped with DiSGro.  
#DATA_DIR = os.environ.get("DISGRO_DATA", "data")
DATA_DIR = files("pydisgro")/"data"


def data_path(name):
    """
    Resolve name against the DiSGro data directory.
    """
    return str((DATA_DIR.joinpath(name)))

FILE_ATOMPROP = "atomProp2.txt"        # atom/residue force-field parameters
FILE_SCTORSION2 = "SCT_PF.txt"         # side-chain torsion angle library
FILE_LOODIS = "LOODIS_ed4_8_V3.txt"    # LOODIS statistical potential
FILE_PARAMETER = "parameter.txt"       # energy-term scaling parameters
FILE_BBT = "BBT_phi_psi_pair_NEW.txt"  # joint phi/psi counts per residue type
FILE_FRAG_N_C = "frag.N.C_pdf_32_19.txt"    # end-to-end distance pdf, label 0
FILE_FRAG_C_CA = "frag.C.CA_pdf_32_19.txt"  # end-to-end distance pdf, label 1
FILE_LOOPGEO = "LoopGeo_37_pdf_21.txt"      # loop geometry (unused, see sampling)

# placeholder template for missing residues
_PLACEHOLDER = ("ATOM  {serial:5d}  H   {resname:>3s} {chain}{resnum:4d}    "
                "   0.000   0.000   0.000  1.00  1.00           H\n")

# ---------------------------------------------------------------------------
# general
# ---------------------------------------------------------------------------

UNDEF = -12345      # "value not set" sentinel, used for atom/residue types
MAX_NUM_RES = 1500
EXPO = 2.718281828  # the C++ uses this truncated e in the Boltzmann factors
PI = math.pi

# ---------------------------------------------------------------------------
# atom / residue geometry
# ---------------------------------------------------------------------------
NUM_BB_ATOM = 6         # N, CA, C, O, H, CB (Gly has a pseudo CB, Pro a pseudo H)
CC_DIS_CUT = 12         # residue centre-centre distance cutoff
RES_CENT_DIS_CUTOFF = 12
CUB_SIZE = 5.5          # slack added to residue radii when testing proximity
VDW_CLASH_CUTOFF = 0.60  # dis/(r1+r2) below this counts as a clash
MAX_HBOND_PER_DONOR = 1
MAX_HBOND_PER_ACCEPTOR = 2
MAX_DISTANCE_HBOND = 3.0
MAX_E_CUTOFF = 10

# ---------------------------------------------------------------------------
# Energy potential
# ---------------------------------------------------------------------------
LOODIS_ATM_TP = 20      # number of heavy-atom types covered by the LOODIS table
LOODIS_DIS_BIN = 80     # 0 - 8 A in 0.1 A bins
H_INLO = 0.1            # LOODIS bin width
H_INTV = 0.1
PF_DIS_CUT = 8          # atom-atom cutoff; beyond this the interaction is ignored
PF_DIS_CUT_SQUARE = 64
LOODIS_PF_DIS_CUT = 15
START_DIS = 1.5
MAX_ENERGY = 10
MAX_SCT_ENERGY = 5
SIMPL_DIS_CUTOFF = 10
SIMPL_INTV = 0.5

B_T_INT = 4             # backbone torsion interval, degrees
SC_T_INT = 4            # side-chain torsion interval, degrees
BBTbinSize = 5          # phi/psi bin width for the joint-angle table, degrees
TORBIN = 360 // BBTbinSize

NUM_RES_TP = 20
MAX_NUM_SC_ST = 50      # max side-chain states per residue
MAX_NUM_STATE = 30
MAX_NUM_MODEL = 5
NUM_BB_ANG = 1000000

# Distance beyond which the LOODIS table is zero.
LOODIS_CUT = H_INLO * LOODIS_DIS_BIN # 8.0 A
LOODIS_CUT_SQ = LOODIS_CUT ** 2

# ---------------------------------------------------------------------------
# index maps replacing the C++ enums
# ---------------------------------------------------------------------------

# map of atom names to int. this also determines the position of each atom in slots allocated per residue
D_AT = {'ATM_N': 0, 'ATM_CA': 1, 'ATM_C': 2, 'ATM_O': 3, 'ATM_H': 4,
        'ATM_CB': 5, 'C_SG': 6, 'D_CG': 6, 'D_OD1': 7, 'D_OD2': 8,
        'E_CD': 7, 'E_OE1': 8, 'E_OE2': 9, 'F_CG': 6, 'F_CD1': 7,
        'F_CD2': 8, 'F_CE1': 9, 'F_CE2': 10, 'F_CZ': 11, 'H_CG': 6,
        'H_ND1': 7, 'H_CD2': 8, 'H_CE1': 9, 'H_NE2': 10, 'I_CG1': 6,
        'I_CG2': 7, 'I_CD1': 8, 'K_NZ': 9, 'Q_CD': 7,}

# Convenience aliases for the backbone slots, which are used constantly.
ATM_N = 0
ATM_CA = 1
ATM_C = 2
ATM_O = 3
ATM_H = 4
ATM_CB = 5
BB_SLOTS = (ATM_N, ATM_CA, ATM_C, ATM_O, ATM_H, ATM_CB)

# Convenience aliases for special residue types.
ALA = 0
GLY = 5
PRO = 12

# map of residue names to int
D_RES = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4, 'GLY': 5,
         'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11,
         'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER': 15, 'THR': 16, 'VAL': 17,
         'TRP': 18, 'TYR': 19}


# maps for energy types and modes, currently only E_LOODIS is in use
ENERGY_TYPES = 25
D_ET = {"E_VDWA": 0, "E_VDWR": 1, "E_VAA": 2, "E_SOL": 3, "E_HALP": 4,
        "E_RP": 5, "E_HBB": 6, "E_HBS": 7, "E_BBT": 8, "E_SCT": 9,
        "E_RAMA": 10, "E_SIMPL": 11, "E_SS": 12, "E_CNT": 13, "E_ROT": 14,
        "E_HAV": 15, "E_RAMAE": 16, "E_RAMAC": 17, "E_CON": 18, "E_EP": 19,
        "E_SRC": 20, "E_MODEL": 21, "E_TEMP": 22, "E_HB": 23, "E_LOODIS": 24}
D_ET = D_ET | {v: k for k, v in D_ET.items()}
E_LOODIS = 24

ENERGY_MODES = 18
D_EM = {"EM_VDW": 0, "EM_SOL": 1, "EM_HALP": 2, "EM_RP": 3, "EM_HB": 4,
        "EM_BBT": 5, "EM_SCT": 6, "EM_RAMA": 7, "EM_SIMPL": 8, "EM_SS": 9,
        "EM_CNT": 10, "EM_ROT": 11, "EM_CON": 12, "EM_EP": 13, "EM_SRC": 14,
        "EM_MODEL": 15, "EM_TEMP": 16, "EM_LOODIS": 17}
D_EM = D_EM | {v: k for k, v in D_EM.items()}
EM_LOODIS = 17

D_PARAM = {"E": 0, "A": 1, "B": 2, "H": 3, "S": 4}

# Global atom-name to index map, ordered as in atomProp2.txt.  Kept for the H
# placement code, which addresses atoms by name.
aiMap = {"N": 0, "CA": 1, "C": 2, "O": 3, "H": 4,
         "ACB": 5, "CCB": 6, "CSG": 7, "DCB": 8, "DCG": 9,
         "DOD1": 10, "DOD2": 11, "ECB": 12, "ECG": 13, "ECD": 14,
         "EOE1": 15, "EOE2": 16, "FCB": 17, "FCG": 18, "FCD1": 19,
         "FCD2": 20, "FCE1": 21, "FCE2": 22, "FCZ": 23, "HCB": 24,
         "HCG": 25, "HND1": 26, "HCD2": 27, "HCE1": 28, "HNE2": 29,
         "HHD1": 30, "HHE2": 31, "ICB": 32, "ICG1": 33, "ICG2": 34,
         "ICD1": 35, "KCB": 36, "KCG": 37, "KCD": 38, "KCE": 39,
         "KNZ": 40, "LCB": 41, "LCG": 42, "LCD1": 43, "LCD2": 44,
         "MCB": 45, "MCG": 46, "MSD": 47, "MCE": 48, "NCB": 49,
         "NCG": 50, "NOD1": 51, "NND2": 52, "N1HD2": 53, "N2HD2": 54,
         "PCB": 55, "PCG": 56, "PCD": 57, "QCB": 58, "QCG": 59,
         "QCD": 60, "QOE1": 61, "QNE2": 62, "Q1HE2": 63, "Q2HE2": 64,
         "RCB": 65, "RCG": 66, "RCD": 67, "RNE": 68, "RCZ": 69,
         "RNH1": 70, "RNH2": 71, "RHE": 72, "R1HH1": 73, "R2HH1": 74,
         "R1HH2": 75, "R2HH2": 76, "SCB": 77, "SOG": 78, "TCB": 79,
         "TOG1": 80, "TCG2": 81, "VCB": 82, "VCG1": 83, "VCG2": 84,
         "WCB": 85, "WCG": 86, "WCD1": 87, "WCD2": 88, "WNE1": 89,
         "WCE2": 90, "WCE3": 91, "WCZ2": 92, "WCZ3": 93, "WCH2": 94,
         "WHH2": 95, "YCB": 96, "YCG": 97, "YCD1": 98, "YCD2": 99,
         "YCE1": 100, "YCE2": 101, "YCZ": 102, "YOH": 103}

# Atom types >= H_ATOM_TYPE are hydrogens/placeholders and are excluded from energy
# 1-20 are heavy atoms, 21 is DNA/RNA phosphorus.
H_ATOM_TYPE = 22

# ---------------------------------------------------------------------------
# loop closure
# ---------------------------------------------------------------------------

CLOSED_N_CA_MIN, CLOSED_N_CA_MAX = 1.358, 1.558
CLOSED_CA_C_MIN, CLOSED_CA_C_MAX = 1.425, 1.625
CLOSED_ANG_NCAC_MIN, CLOSED_ANG_NCAC_MAX = 86.1, 136.1
CLOSED_ANG_CACN_MIN, CLOSED_ANG_CACN_MAX = 95.0, 135.0
CLOSED_OMEGA_MIN = 160.0  # |torsion| must exceed this

# max polinomials
MAX_SOLN = 16
DEG_POL = 16

# ---------------------------------------------------------------------------
# SMC parameters
# ---------------------------------------------------------------------------

# Number of distance proposals drawn before the Ramachandran reweighting step.
LARGE_NUM_DISTANCE_STATES = 160
# Conformations further above the running minimum than this are not stored.
ENERGY_CUTOFF = 350.0
# Hard caps the C++ applies to the stored ensemble.
MAX_STORED_LOOPS = 10000
MAX_CLOSED_CONF = 5000
# Retries of the analytic closure before a trial is abandoned.
MAX_CLOSURE_RETRIES = 300
CLOSURE_JITTER = (0.06, PI / 32, PI / 32)
# A loop residue with more clashes than this rejects the whole conformation.
MAX_CLASH_PER_RESIDUE = 10

# ---------------------------------------------------------------------------
# sampling parameters
# ---------------------------------------------------------------------------

# Number of fragment lengths and distance bins in the empirical tables.
MAX_FRAG_LEN = 20
N_DIST_BIN = 32
