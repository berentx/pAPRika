#!/bin/bash
#SBATCH --ntasks 1

cpptraj -i cpptraj.in

MMPBSA.py -i mmpbsa.in -o mmpbsa.dat -cp complex/vac.prmtop -sp _prmtop_ -y mdcrd -rp complex/receptor.prmtop -lp complex/ligand.prmtop

