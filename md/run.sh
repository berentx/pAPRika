#!/bin/bash

path=$(dirname "$0")

echo "submit init job"
init=$(sbatch --parsable "${path}/init.sbatch")

echo "submit run job"
equil=$(sbatch --parsable --dependency=afterok:${init} "${path}/run.sbatch")

