# init file to run as python module with -m

from .constants import DATA_DIR, data_path
from .geom import Atom, Point, seed
from .potential import PF
from .residue import Residue, Rotamer, SCR
from .structure import flatStructure, Root_MSD, Structure

__all__ = [
    "Atom", "flatStructure", "PF", "Point", "Residue", "Root_MSD", "Rotamer",
    "SCR", "Structure", "DATA_DIR", "data_path", "init_parameters",
    "load_structure", "seed",]

def init_parameters(sidechains=True, verbose=False):
    """
    Initialze the Atom, Residue and Potential classes by loading in
    the neccesary parameter tables. Must be called once 
    before building or sampling any structure.
    """
    Atom.InitPar(verbose=verbose)
    Residue.InitMap()
    Residue.InitPar(verbose=verbose)
    PF.InitPar(verbose=verbose)
    PF.InitLOODIS(verbose=verbose)
    if sidechains:
        SCR.InitSCAng(verbose=verbose)

def load_structure(path, add_hydrogens=False):
    """
    Read a PDB file into a Structure object
    """
    struct = Structure.readPdb(path)
    if add_hydrogens:
        struct.addH()
    return struct

# initialize parameters when loading package
init_parameters()