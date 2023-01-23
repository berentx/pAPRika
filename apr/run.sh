#!/bin/bash

path=$(dirname "$0")

echo "submit init job"
init=$(sbatch --parsable "${path}/init.sbatch")

echo "submit equil job"
equil=$(sbatch --parsable --dependency=afterok:${init} "${path}/equil.sbatch")

#echo "submit run job"
#find windows/ -mindepth 1 -maxdepth 1 -type d | sort > window_list.dat
#num_windows=$(wc -l window_list.dat | awk '{print $1}')
#run=$(sbatch --parsable --dependency=afterok:${equil} --array=1-${num_windows} "${path}/run.sbatch")
#
#echo "submit analysis job"
#analysis=$(sbatch --parsable --dependency=afterok:${run} "${path}/analysis.sbatch")
#
