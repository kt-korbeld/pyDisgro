import argparse
import os
import sys

from . import constants
from .geom import seed
from .potential import PF
from .residue import Residue, SCR
from .smc import SMC
from .structure import Root_MSD, Structure
from .structure import resnum_to_index, blank_loop, renumber_pdb

def build_parser(advanced_help=False):
    # define function that only prints if advanced_help is true
    def adv_help(description):
        return description if advanced_help else argparse.SUPPRESS
    
    p = argparse.ArgumentParser( prog="discgro",
        description="Distance-guided sequential chain-growth Monte Carlo loop modelling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    basic = p.add_argument_group("Basic Arguments")
    basic.add_argument("-f", "--protfile", required=True, help="input protein coordinate file (.pdb)")
    basic.add_argument("-s", "--start", type=int, required=True, help="PDB residue number of the N-terminal loop anchor")
    basic.add_argument("-e", "--end", type=int, required=True, help="PDB residue number of the C-terminal loop anchor")
    basic.add_argument("-l", "--loopseq", type=str, default=None, help="one-letter sequence of the residues between the anchors")
    basic.add_argument("-ch", "--chain", default=None, help="chain the anchors belong to (only needed when the residue numbers are ambiguous)")
    basic.add_argument("-o", "--outdir", default="pdb_output", help="directory for output PDB files")

    advanced = p.add_argument_group("Advanced Arguments")
    advanced.add_argument("-m", "--mode", default="smc", choices=["smc"], help=adv_help("mode of computation"))
    advanced.add_argument("-n", "--num_conf", type=int, default=5000, help=adv_help("number of loop conformations to attempt"))
    advanced.add_argument("-nds", "--ndist", default="32", help=adv_help("number of distance sampling states (comma sep for per-residue)"))
    advanced.add_argument("-nas", "--nangs", default="0", help=adv_help("number of angle sampling states"))
    advanced.add_argument("-cfk", "--confkeep", type=int, default=1, help=adv_help("number of retained conformations"))
    advanced.add_argument("-nsc", "--nscc", type=int, default=0, help=adv_help("number of side chain states (0 disables side chains)"))
    advanced.add_argument("-nout", "--pdbout", type=int, default=1, help=adv_help("number of conformations to write to pdb_output/"))
    advanced.add_argument("-eval", "--eval", action="store_true", help=adv_help("re-evaluate energies after side chains are placed"))
    advanced.add_argument("-cl", "--close", type=int, default=1, help=adv_help("use analytic closure (1) or not (0)"))
    advanced.add_argument("-nsco", "--noscore", action="store_true", help=adv_help("do not score or store conformations"))
    advanced.add_argument("-t", "--temperature", type=float, default=1.0, help=adv_help("temperature for the side chain Boltzmann weights"))
    advanced.add_argument("-vdw", "--vdw_adj", type=float, default=1.0, help=adv_help("van der Waals radius adjustment"))
    advanced.add_argument("-agt", "--ang_type", type=int, default=2, help=adv_help("side chain torsion representation type"))
    advanced.add_argument("-prn", "--protname", default='', help=adv_help("override the protein name taken from the file name"))
    advanced.add_argument("-dd", "--datadir", default=None, help=adv_help("directory holding the DiSGro parameter files"))
    advanced.add_argument("-sd", "--seed", type=int, default=None, help=adv_help("random seed (omit for a nondeterministic run)"))
    advanced.add_argument("-rmsd", "--rmsdout", default=None, help=adv_help("write per-conformation loop RMSD to this file; needs -native"))
    advanced.add_argument("-nat", "--native", default=None, help=adv_help("reference PDB with the loop present, for RMSD"))
    advanced.add_argument("-q", "--quiet", action="store_true", help=adv_help("suppress progress output"))

    # raise exception for settings removed from the original C++ port
    for dead in ("--ellip", "--kcluster", "--Surface", "--refine", "--selres", "-d-umpseq"):
        p.add_argument(dead, action=_Unsupported, nargs=0 if dead in
                       ("--ellip", "--kcluster") else "?",
                       help=argparse.SUPPRESS)
    # add advanced help
    p.add_argument("-h+", "--advanced_help", action="store_true", help="show advanced arguments")
    return p

# deal with unsupported arguments
class _Unsupported(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} is not implemented in this port "
                     "(see the module docstring for the omitted features)")

# call advanced help
class AdvancedHelpAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        make_parser(advanced_help=True).print_help()
        parser.exit()

def _states(text):
    return tuple(int(v) for v in text.split(","))


def main(argv=None):
     # if advanced help is enabled, print all options
    advanced_help = False
    for arg in sys.argv:
        if arg == "--advanced-help" or arg == "-h+":
            advanced_help = True
    if advanced_help:
        build_parser(advanced_help=True).print_help()
        raise SystemExit
    # parse arguments
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # return advanced help
    if args.advanced_help:
        make_parser(advanced_help=True).print_help()
        raise SystemExit
    # set data dir
    if args.datadir:
        constants.DATA_DIR = args.datadir
    seed(args.seed)

    from .geom import Atom
    Atom.vdw_adj = args.vdw_adj
    Atom.InitPar()
    Residue.InitMap()
    Residue.InitPar()
    PF.InitPar()
    PF.InitLOODIS()
    if args.nscc > 0:
        SCR.InitSCAng()

    name = args.protname or os.path.basename(args.protfile).rsplit(".", 1)[0]

    # prepare input pdb with placeholder atoms if loopseq is specified
    if args.loopseq:
        # read pdb for lines
        with open(args.protfile) as f:
            lines = f.readlines()
        # construct pdb with placeholder atoms
        lines = blank_loop(lines, args.start, args.end, args.loopseq, args.chain)
        # save new input pdb in output directory
        os.makedirs(args.outdir, exist_ok=True)
        loop_input = os.path.join(args.outdir, f"{name[:4]}_{args.start}_{args.end}_loopinput.pdb")
        with open(loop_input, "w") as f:
            f.writelines(lines)
        conf = Structure.readPdb(lines)
        # report result if verbose
        if not args.quiet:
            print(f"Blanked {len(args.loopseq.strip())} residues between "
                  f"{args.start} and {args.end}; wrote {loop_input}")
    else:
        conf = Structure.readPdb(args.protfile)
    conf._ProtName = name

    # --start/--end are PDB residue numbers; SMC indexes by position.
    try:
        start = resnum_to_index(conf, args.start, args.chain)
        end = resnum_to_index(conf, args.end, args.chain)
    except ValueError as exc:
        parser.error(str(exc))
    if end <= start:
        parser.error(f"--end ({args.end}) does not come after --start "
                     f"({args.start}) in {conf._ProtName}")
    if end - start < 2:
        parser.error(f"there are no residues between {args.start} and "
                     f"{args.end} in {conf._ProtName} to build; the loop "
                     f"residues have to be in the file as H placeholders, or "
                     f"be supplied with --loopseq")

    if not args.quiet:
        print(f"Protein Name:        {conf._ProtName}")
        print(f"Start Residue  {args.start}   :   End Residue  {args.end}"
              f"   (positions {start} : {end}, {end - start - 1} residues to build)")

    smc = SMC(conf, start, end,
              num_conf=args.num_conf,
              num_distance_states=_states(args.ndist),
              num_angle_states=_states(args.nangs),
              confkeep=args.confkeep,
              sample_sc=args.nscc > 0,
              num_sc_states=max(args.nscc, 1),
              ang_type=args.ang_type,
              evaluate=args.eval,
              close=bool(args.close),
              no_score=args.noscore,
              temperature=args.temperature,
              verbose=not args.quiet)
    results = smc.run()

    if args.pdbout > 0 and results:
        os.makedirs(args.outdir, exist_ok=True)
        for i, r in enumerate(results[:args.pdbout]):
            out = smc.to_structure(r)
            path = os.path.join(
                args.outdir,
                f"{name[:4]}_{args.start}_{args.end}_{i + 1}topconf.pdb")
            if args.nscc > 0:
                out.writePdb(path, 1, out.numRes)
            else:
                # No side chains were sampled for the loop, so write it
                # backbone-only rather than emitting stale coordinates.
                out.writePdb(path, 1, out.numRes, start, end)
        if not args.quiet:
            print(f"Wrote {min(args.pdbout, len(results))} conformations to "
                  f"{args.outdir}/")

    if args.rmsdout:
        if not args.native:
            parser.error("-rmsdout requires -native")
        native = Structure.readPdb(args.native)
        # The reference structure is numbered like the input but need not be
        # indexed like it, so translate its anchors separately.
        try:
            nat_start = resnum_to_index(native, args.start, args.chain)
        except ValueError as exc:
            parser.error(str(exc))
        with open(args.rmsdout, "w") as f:
            for i, r in enumerate(results):
                out = smc.to_structure(r)
                rms = Root_MSD(native, out, nat_start,
                               nat_start + (end - start), start)
                f.write(f"{i}, {rms:.4f}\n")
        if not args.quiet:
            print(f"Wrote loop RMSD for {len(results)} conformations to "
                  f"{args.rmsdout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
