#!/bin/bash
#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 2

env

python /shared/compounds/pAPRika/apr/cli.py init --config config.yml
#python /shared/compounds/pAPRika/apr/cli.py init --complex $complex --host $host --guest $guest --g1 $g1 --g2 $g2 --r0 $r0 --r1 $r1

