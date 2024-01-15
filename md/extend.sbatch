#!/bin/bash
#SBATCH --gpus 1
#SBATCH --ntasks 1

. "/home/sunhwan/miniconda3/etc/profile.d/conda.sh"
conda activate beren

python /shared/compounds/pAPRika/md/cli.py run --ns 100 --extend

