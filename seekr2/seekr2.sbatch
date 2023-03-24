#!/bin/bash
#SBATCH --partition g5
#SBATCH --gpus 1

source /shared/miniconda3/etc/profile.d/conda.sh 
conda activate SEEKR
export LD_LIBRARY_PATH="/shared/apps/seekr2/openmm/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/shared/apps/seekr2/openmm/lib/plugins:$LD_LIBRARY_PATH"

if [ ! -f seekr2/model.xml ]; then
  python /shared/compounds/pAPRika/seekr2/cli.py init --config config.yml
  python /shared/apps/seekr2/seekr2/seekr2/prepare.py -s seekr2.xml
fi

python /shared/apps/seekr2/seekr2/seekr2/run.py -t 25000000 any seekr2/model.xml
python /shared/apps/seekr2/seekr2/seekr2/analyze.py -s seekr2/model.xml > seekr2.dat
python /shared/apps/seekr2/seekr2/seekr2/converge.py -s seekr2/model.xml

