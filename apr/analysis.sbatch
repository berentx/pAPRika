#!/bin/bash
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 2

env

python /shared/compounds/pAPRika/apr/cli.py analysis > analysis.dat

