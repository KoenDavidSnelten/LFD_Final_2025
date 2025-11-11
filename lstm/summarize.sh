#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --partition=regular
#SBATCH --mem=2G
#SBATCH --output=output/bert/slurm-%A_%a.out
#SBATCH --array=0-7

# --- 1. Load Modules and Activate Environment ---
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0

source $HOME/venvs/venv_lfd_final/bin/activate

# --- 2. Define Variables and Select Array Element ---
# Array of preprocessing steps
PRE_STEPS=("lower_case" "no_emoji" "remove_hastag" "remove_html" "remove_url" "no_emoji_remove_contractions" "remove_contractions" "remove_swear")

# Select the current preprocessing step using the SLURM_ARRAY_TASK_ID
# ${#PRE_STEPS[@]} returns the size of the array (8). 
# The modulus operator (%) ensures the index wraps around if the array size is smaller than the array range (optional, but good practice).
PRE_STEP=${PRE_STEPS[$SLURM_ARRAY_TASK_ID % ${PRE_STEPS[@]}]}

# Define the base output directory for *this specific step*
# You should typically use $SLURM_USER instead of $USER in a slurm environment
BASE_OUTPUT_DIR="/scratch/$USER/LFD_Final_2025/lm/output/bert/$PRE_STEP"


# Define the full path to the results file for *this specific step*
# The script summarizes outputs from subdirectories, it shouldn't write to a RESULTS_FILE 
# *inside* the job, it should use the individual run directories inside BASE_OUTPUT_DIR.
# If you intend to write the output of the summarizer *itself* to a file, 
# you should define a location outside of the run directories, like:
SUMMARY_OUTPUT_FILE="/scratch/$USER/LFD_Final_2025/lm/bert/${PRE_STEP}_summary.txt"


# --- 3. Execute Script ---
echo "Job starting for pre-step: $PRE_STEP" > "$RESULTS_FILE"
echo "Searching in base directory: $BASE_OUTPUT_DIR" >> "$RESULTS_FILE"

# Your python script is run, passing the PRE_STEP as the --path argument, 
# and the output is redirected to the $SUMMARY_OUTPUT_FILE.
python3 summarize_outputs.py \
  --path "$PRE_STEP" \
  >> "$SUMMARY_OUTPUT_FILE" 2>&1

echo "Job ending"

# --- 4. Deactivate Environment ---
deactivate