#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=regular
#SBATCH --mem=4000
#SBATCH --output=./output/%x_%j.out


# Load correct Python and CUDA versions
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

# Activate your virtual environment
source $HOME/venvs/venv_lfd_final/bin/activate


echo "Running Job"

python3 svm_model.py 

echo "Ending Job"
deactivate