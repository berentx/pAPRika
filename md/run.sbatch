#!/bin/bash
##SBATCH --partition g5
#SBATCH --gpus 1
#SBATCH --ntasks 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate beren

python /shared/compounds/pAPRika/md/cli.py run --ns 50

