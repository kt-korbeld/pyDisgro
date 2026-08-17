# pyDisgro
A numpy-only Python port of [DiSGro](https://doi.org/10.1371/journal.pcbi.1003539)
(Tang, Zhang & Liang, *PLoS Computational Biology* 2014) — distance-guided
sequential chain-growth Monte Carlo for reconstructing missing protein loops.

## Installation
pyDisgro is a minimal port that only requires Python 3.9+ and numpy. It can be installed using:
```
pip install pydisgro
```
 ## Using the Disgro CLI
 Disgro can be easily run using the CLI interface:
 ```
disgro -f tests/4p79.pdb -s 33 -e 42 -l VHGNVITT
```
Where `-f` or `--protfile` specifies the input PDB, `-s` or `--start` and `-e` or `--end` the PDB residue numbers of the two loop anchors, and `-l` or `--loopseq` the sequence of the loop to be inserted. 

Note that the numbering for the `--start` and `--end` input parameters are different from the original C++ Disgro application, where a 0-indexed residue index rather than the PDB residue numbers were used.

The following advanced settings are available, which correspond to the original DisGro flags. In most cases, these do not have to be adapted. A complete overview of the flags can also be accessed using `-h+` or `--advanced-help`. 
```
--protfile      input protein coordinate file (.pdb)
--start         PDB residue number of the N-terminal loop anchor
--end           PDB residue number of the C-terminal loop anchor
--loopseq       one-letter sequence of the residues between the anchors
--chain         Chain the anchors belong to (only needed when the residue numbers are ambiguous)       
--outdir        Directory for output PDB files (default: pdb_output)
--mode          Mode of computation. Current implementation only has SMC as an option (default:smc)
--num_conf      Number of loop conformations to attempt. (default: 5000)
--ndist         Number of distance sampling states (comma separated for per-residue). (default: 32)
--nangs         Number of angle sampling states (default: 0)
--confkeep      Number of retained conformations (default: 1)
--nscc          Number of side chain states (0 disables side chains)
--pdbout        Number of conformations to write to output directory (default: 1)
--eval          Re-evaluate energies after side chains are placed
--close         Use analytic closure (1) or not (0) (default: 1)
--noscore       Do not score or store conformations
--temperature   Temperature for the side chain Boltzmann weights (default: 1.0)
--vdw_adj       Van der Waals radius adjustment (default: 1Å)
--ang_type      Side chain torsion representation type (default: 2)
--protname      Override the protein name taken from the file name
--seed          Random seed (omit for a nondeterministic run)
--rmsdout       Write per-conformation loop RMSD to this file (needs --native)
--native        Reference PDB with the loop present, for RMSD
--quiet         suppress progress output
```

## Using Disgro as a python package
Disgro can also be integrated into other pipelines as a python library:
```python
import disgro
from disgro.smc import SMC

# initialize Disgro and load input PDB
disgro.init_parameters()
conf = disgro.load_structure("1ctqa.pdb")

# initialize sequential Monte Carlo with selected settings
smc = SMC(conf, start=start, end=end, num_conf=5000, confkeep=100, sample_sc=True, num_sc_states=20, evaluate=True)

# run sequential Monte Carlo, and save best loop structure as PDB
loops = smc.run()
best = smc.to_structure(loops[0]) 
best.writePdb("loop.pdb", 1, best.numRes)
```
Without the CLI wrapper, the SMC solver assumes the original Disgro input format, meaning the `start=` and `end=` variables are the 0-indexed residue index, and the input structure requires placeholder H atoms with `[0,0,0]` coordinates for the missing loop residues. The structure with added placeholder atoms and the appropriate residue index can be prepared using:
```python
import disgro
from disgro.structure import resnum_to_index, blank_loop, renumber_pdb

disgro.init_parameters()

# open pdb file as list of strings, and add placeholder atoms for loop residues
with open("1ctqa.pdb") as f:
    pdblines = f.readlines()
    pdblines_loop = blank_loop(pdblines, 33, 42, "VHGNVITT", "A")

# load adapted pdb lines into a structure and convert to residue index
conf = disgro.load_structure(pdblines_loop)
start, end = resnum_to_index(conf, 33), resnum_to_index(conf, 42)
```
