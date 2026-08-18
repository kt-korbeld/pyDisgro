# classes for saving variables needed for potential function (PF)
# architecture was meant for more energy types, but currently only EM_LOODIS is implemented

import os
import numpy as np
from .constants import *

class PF:
    """
    Class with static variables for potential-function parameters.
    """
    useBfactor = 0
    contWt = 0
    T = 1.0
    COUNT = True
    intercept = 0.0

    # Which energy modes are switched on.  main.cpp enables EM_LOODIS only.
    cal = [i == EM_LOODIS for i in range(ENERGY_MODES)]

    # contains 20x20 atom interaction types. set to 21x21 for security
    # so that nucleic-acid phosphorus (type 21) cannot index out of bounds
    LOODIS = np.zeros((21, 21, LOODIS_DIS_BIN))

    Parameter = np.zeros((5, ENERGY_TYPES))

    @classmethod
    def InitPar(cls, parFile=None, verbose=False):
        """
        Read the per-term scaling parameters from parameter.txt.
        """
        # sanity checks
        if parFile is None:
            parFile = data_path(FILE_PARAMETER)
        if verbose:
            print("Reading Energy Parameters in ", parFile)
        if not os.path.exists(parFile):
            raise FileNotFoundError(f"cannot open parameter file {parFile}")
        # initialize output parameters and go over each line in file
        cls.Parameter = np.zeros((5, ENERGY_TYPES))
        with open(parFile) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                key, val = line.strip().split(":")
                group, index, value = key[0], int(key[1:]), float(val)
                if value != 0:
                    cls.Parameter[D_PARAM[group], index] = value

    @classmethod
    def InitLOODIS(cls, filename=None, verbose=False):
        """
        Read the LOODIS atom-atom distance potentials. for each atom type
        Atom types in the file are 0-based, so an atom of ._type t indexes row t - 1
        """
        # sanity checks
        if filename is None:
            filename = data_path(FILE_LOODIS)
        if verbose:
            print("Reading LOODIS Parameters in ", filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(f"cannot open parameter file {filename}")
        # initialize output parameters and go over each line in file
        table = np.zeros((21, 21, LOODIS_DIS_BIN))
        with open(filename) as f:
            for line in f:
                # skip commented lines, short lines, or lines missing data.
                if not line or line[0] == "#" or len(line) < 3:
                    continue
                linedata = line.split()
                if len(linedata) != 6:
                    continue
                k, i, j = int(linedata[0]), int(linedata[1]), int(linedata[2])
                score = float(linedata[5])
                table[i, j, k] = score
                table[j, i, k] = score
        cls.LOODIS = table

    @classmethod
    def set_modes(cls, *modes):
        """
        Enable exactly the listed energy modes (by index or name).
        """
        cls.cal = [False] * ENERGY_MODES
        for m in modes:
            cls.cal[D_EM[m] if isinstance(m, str) else m] = True
