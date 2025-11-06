#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --output=output/slurm-%A_%a.out


module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/venv_lfd_final/bin/activate

echo "Job starting"

python3 lm_model.py

echo "Job ending"

deactivate