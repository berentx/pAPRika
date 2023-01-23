#!/bin/bash
#SBATCH --partition cpu
#SBATCH --ntasks 8

echo $parent $name
cd $parent/$name

cp ../../cpptraj.in ../../mmpbsa.in .

cpptraj -i cpptraj.in

MMPBSA.py -i mmpbsa.in -o mmpbsa.dat -cp complex/vac.prmtop -sp windows/1/system-solvated.prmtop -y windows/1/processed.mdcrd -rp complex/receptor.prmtop -lp complex/ligand.prmtop

