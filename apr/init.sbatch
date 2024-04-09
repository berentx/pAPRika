#!/bin/bash
##SBATCH --partition c5a
#SBATCH --nodes 1
#SBATCH --ntasks 4

env

python /shared/compounds/pAPRika/apr/cli.py init --config config.yml
#python /shared/compounds/pAPRika/apr/cli.py init --complex $complex --host $host --guest $guest --g1 $g1 --g2 $g2 --r0 $r0 --r1 $r1

