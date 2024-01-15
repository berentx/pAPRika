import argparse
import logging
import os
from pathlib import Path
import shutil

from init import init
from run import run
from analysis import analysis

logger = logging.getLogger("md")
logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description='setup parprika simulations', add_help=False)
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    subparsers = parser.add_subparsers(help='sub-command help')

    # create the parser for the "init" command
    parser_init = subparsers.add_parser('init', help='init help')
    parser_init.add_argument('--config', help='config file')
    parser_init.add_argument('--host', help='host mol2 file')
    parser_init.add_argument('--guest', help='guest mol2 file')
    parser_init.add_argument('--complex', help='complex PDB file')
    parser_init.add_argument('--copy', type=int, default=1, help='copy solutes')
    parser_init.add_argument('--implicit', action='store_true', help='use implicit solvent (default: explicit solvent)')
    parser_init.add_argument('--overwrite', action='store_true', help='force overwrite')
    parser_init.add_argument('--conc', type=float, default=50.0, help="ion concentration (mM)")
    parser_init.add_argument('--nwater', type=int, default=1000)
    parser_init.set_defaults(func=init)

    # create the parser for the "run" command
    parser_run = subparsers.add_parser('run', help='run help', parents=[parser])
    parser_run.add_argument('--implicit', action='store_true', help='use implicit solvent (default: explicit solvent)')
    parser_run.add_argument('--ns', type=float, default=1)
    parser_run.add_argument('--extend', action='store_true')
    parser_run.set_defaults(func=run)

    # create the parser for the "analysis" command
    parser_analysis = subparsers.add_parser('analysis', help='analysis help', parents=[parser])
    parser_analysis.add_argument('--host', help='host mol2 file')
    parser_analysis.set_defaults(func=analysis)

    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    args.func(args)
    """
    try:
        args.func(args)
    except Exception as e:
        print(e)
        parser.print_help()
    except:
        parser.print_help()
    """


if __name__ == '__main__':
    main()
