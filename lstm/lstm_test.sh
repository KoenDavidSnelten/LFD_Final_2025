#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --mem=8G
#SBATCH --output=output/slurm-%A_%a.out


source $HOME/venvs/venv_lfd_final/bin/activate

export CUDA_VISIBLE_DEVICES=-1


echo "Job starting"

python3 lstm_model.py \
    --verbose 0

echo "Job ending"

deactivate