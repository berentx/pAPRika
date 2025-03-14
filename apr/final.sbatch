#!/bin/bash
#SBATCH --gpus 1
#SBATCH --ntasks 1

python /shared/compounds/pAPRika/apr/cli.py run --ns 10 --final

