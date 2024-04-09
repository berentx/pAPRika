#!/bin/bash
##SBATCH --partition g5
#SBATCH --gpus 1
#SBATCH --ntasks 1

python /shared/compounds/pAPRika/apr/cli.py run --ns 40 --final

