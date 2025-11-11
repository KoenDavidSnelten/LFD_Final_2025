#!/bin/bash
#SBATCH --job-name=setup_env
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --partition=regular

# Load correct Python and CUDA versions
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

# Activate your virtual environment
source $HOME/venvs/venv_lfd_final_lm/bin/activate

pip install torch torchvision torchaudio

deactivate

