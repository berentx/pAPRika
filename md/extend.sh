#!/bin/bash

path=$(dirname "$0")

echo "submit run job"
equil=$(sbatch --parsable --dependency=afterok:${init} "${path}/extend.sbatch")
