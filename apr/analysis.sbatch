#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate beren

path=/shared/compounds/pAPRika/

# apr
python ${path}/apr/cli.py analysis > apr.dat

# mmpbsa
prmtop=windows/a000/apr-solvated.prmtop
dcd=windows/a000/production.dcd
mdcrd=windows/a000/processed.mdcrd

sed -e "s/prmtop/${prmtop//\//\\/}/g" \
    -e "s/dcd/${dcd//\//\\/}/g" \
    -e "s/mdcrd/${mdcrd//\//\\/}/g" \
    ${path}/mmpbsa/cpptraj.in > cpptraj.in

cpptraj -i cpptraj.in

cp ${path}/mmpbsa/mmpbsa.in .

MMPBSA.py -i mmpbsa.in -o mmpbsa.dat -cp complex/vac.prmtop \
	  -sp $prmtop -y $mdcrd \
	  -rp complex/receptor.prmtop -lp complex/ligand.prmtop > mmpbsa.dat

rm -f ${mdcrd} _MMPBSA*

