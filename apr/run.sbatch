#!/bin/bash
##SBATCH --partition g5
#SBATCH --gpus 1
#SBATCH --ntasks 1

taskid=$SLURM_ARRAY_TASK_ID
window=$(sed -n ${taskid}p window_list.dat | xargs basename)

if [ ! -f $window/production.pdb ]; then
  python /shared/compounds/pAPRika/apr/cli.py run --ns 40 --window $window
fi

