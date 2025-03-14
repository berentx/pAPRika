#!/bin/bash
#SBATCH --gpus 1
#SBATCH --ntasks 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate beren

taskid=$SLURM_ARRAY_TASK_ID
window=$(sed -n ${taskid}p window_list.dat | xargs basename)

if [ ! -f $window/production.pdb ]; then
  python /shared/compounds/pAPRika/apr/cli.py run --ns 10 --window $window
fi

