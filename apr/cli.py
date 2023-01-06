import argparse
import logging
import os
from pathlib import Path
import shutil

from init import *
from run import *
from analysis import *

logger = logging.getLogger("apr")
logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='setup parprika simulations', add_help=False)
    parser.add_argument('--verbose', action='store_true', help='complex PDB file')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "init" command
    parser_init = subparsers.add_parser('init', help='init help')
    parser_init.add_argument('--config', help='config file')
    parser_init.add_argument('--complex', help='complex PDB file')
    parser_init.add_argument('--host', help='host mol2 file')
    parser_init.add_argument('--guest', help='guest mol2 file')
    parser_init.add_argument('--implicit', action='store_true', help='use implicit solvent (default: explicit solvent)')
    parser_init.add_argument('--overwrite', action='store_true', help='force overwrite')
    parser_init.add_argument('--h1', help='H1 anchor atom name')
    parser_init.add_argument('--h2', help='H2 anchor atom name')
    parser_init.add_argument('--h3', help='H3 anchor atom name')
    parser_init.add_argument('--g1', help='G1 anchor atom name')
    parser_init.add_argument('--g2', help='G2 anchor atom name')
    parser_init.add_argument('--d0', type=float, default=-6.0, help="dummy anchor position")
    parser_init.add_argument('--r0', type=float, default=6.0, help="initial distance")
    parser_init.add_argument('--r1', type=float, default=28.0, help="final distance")
    parser_init.add_argument('--offset', type=float, default=0.5, help="distance offset")
    parser_init.add_argument('--k_dist', type=float, default=6.0, help="force constance for distance restraint")
    parser_init.add_argument('--conc', type=float, default=10.0, help="ion concentration (mM)")
    parser_init.add_argument('--nwater', type=int, default=3500)
    parser_init.add_argument('--host_resname', required=False, help='host residue name (default: MOL)')
    parser_init.add_argument('--host_resnr', required=False, help='host residue number (default: None)')
    parser_init.add_argument('--guest_resname', required=False, help='guest residue name (default: LIG)')
    parser_init.add_argument('--guest_resnr', required=False, help='guest residue number (default: None)')
    parser_init.set_defaults(func=init)

    # create the parser for the "run" command
    parser_run = subparsers.add_parser('run', help='run help', parents=[parser])
    parser_run.add_argument('--config', help='config file')
    parser_run.add_argument('--implicit', action='store_true', help='use implicit solvent (default: explicit solvent)')
    parser_run.add_argument('--ns', type=float, default=1)
    parser_run.add_argument('--window', type=str, default="all")
    parser_run.add_argument('--extend', action='store_true')
    parser_run.add_argument('--equilibrate', action='store_true')
    parser_run.set_defaults(func=run)

    # create the parser for the "analysis" command
    parser_analysis = subparsers.add_parser('analysis', help='analysis help', parents=[parser])
    parser_analysis.add_argument('--config', help='config file')
    parser_analysis.set_defaults(func=analysis)

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    args.func(args)


if __name__ == '__main__':
    main()
