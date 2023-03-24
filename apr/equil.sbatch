#!/bin/bash
#SBATCH --partition g5
#SBATCH --gpus 1
#SBATCH --ntasks 1

path=/shared/compounds/pAPRika/apr

echo "submit run job"
find windows/ -mindepth 1 -maxdepth 1 -type d | sort > window_list.dat
num_windows=$(wc -l window_list.dat | awk '{print $1}')
run=$(sbatch --parsable --dependency=afterok:${SLURM_JOB_ID} --array=1-${num_windows} "${path}/run.sbatch")

echo "submit run-final job"
final=$(sbatch --parsable --dependency=afterok:${run} "${path}/final.sbatch")

echo "submit analysis job"
analysis=$(sbatch --parsable --dependency=afterok:${final} "${path}/analysis.sbatch")

python /shared/compounds/pAPRika/apr/cli.py run --equilibrate

