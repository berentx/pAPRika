#!/bin/bash
#SBATCH --partition g5
#SBATCH --gpus 1
#SBATCH --ntasks 1

python /shared/compounds/pAPRika/md/cli.py run --ns 100 

