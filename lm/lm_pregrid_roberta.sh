#!/bin/bash
#SBATCH --job-name=roberta_offensive
#SBATCH --output=/scratch/%u/LFD_Final_2025/lm/output/roberta/slurm_logs/slurm-%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-8

# Basic setup
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Setup modules and python
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
source $HOME/venvs/venv_lfd_final_lm/bin/activate


# Define the hyperparam list
LRS=(2e-5 3e-5 5e-5)
BATCH_SIZES=(32 32 64)
MAX_LENGTHS=(256 64 64)
PATIENCE_VALUES=(2 3 4)
PRE_STEPS=("no_emoji_remove_contractions" "remove_contractions" "remove_swear")

# Derived counts
NUM_HPARAMS=${#LRS[@]}   # 3
NUM_PRESTEPS=${#PRE_STEPS[@]}  # 4

# Compute indices
HPARAM_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_HPARAMS ))
PRESTEP_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_HPARAMS ))

# Select current params
PRE_STEP=${PRE_STEPS[$PRESTEP_IDX]}
LR=${LRS[$HPARAM_IDX]}
BATCH_SIZE=${BATCH_SIZES[$HPARAM_IDX]}
MAX_LENGTH=${MAX_LENGTHS[$HPARAM_IDX]}
PATIENCE=${PATIENCE_VALUES[$HPARAM_IDX]}

# Paths
DATA_DIR="/scratch/$USER/LFD_Final_2025/data/$PRE_STEP"
TRAIN_FILE="$DATA_DIR/train.tsv"
DEV_FILE="$DATA_DIR/dev.tsv"
TEST_FILE="$DATA_DIR/test.tsv"
MODEL="roberta-base"
EPOCHS="15"
BASE_OUTPUT_DIR="/scratch/$USER/LFD_Final_2025/lm/output/roberta"

# Output directory
OUTPUT_DIR="$BASE_OUTPUT_DIR/${PRE_STEP}/lr_${LR}__bs_${BATCH_SIZE}__len_${MAX_LENGTH}__pat_${PATIENCE}"
mkdir -p "$OUTPUT_DIR"

# Results file
RESULTS_FILE="$OUTPUT_DIR/results.txt"

# Log configuration
echo "--- Grid Search Run ---" > "$RESULTS_FILE"
echo "Job ID: $SLURM_ARRAY_TASK_ID" >> "$RESULTS_FILE"
echo "Preprocessing Step: $PRE_STEP" >> "$RESULTS_FILE"
echo "Model: $MODEL" >> "$RESULTS_FILE"
echo "Learning Rate: $LR" >> "$RESULTS_FILE"
echo "Batch Size: $BATCH_SIZE" >> "$RESULTS_FILE"
echo "Max Epochs: $EPOCHS" >> "$RESULTS_FILE"
echo "Max Length: $MAX_LENGTH" >> "$RESULTS_FILE"
echo "Patience: $PATIENCE" >> "$RESULTS_FILE"
echo "Output Dir: $OUTPUT_DIR" >> "$RESULTS_FILE"
echo "-----------------------" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "--- Training Log ---" >> "$RESULTS_FILE"

# Run training
python lm_model.py \
    --model_name "$MODEL" \
    --train_file "$TRAIN_FILE" \
    --dev_file "$DEV_FILE" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --max_length "$MAX_LENGTH" \
    --early_stopping_patience "$PATIENCE" \
    --load_best_model_at_end \
    --metric_for_best_model "eval_f1_macro" \
    --eval_strategy "epoch" \
    >> $RESULTS_FILE 2>&1

echo "Job finished. Results and model saved to $OUTPUT_DIR"

deactivate
