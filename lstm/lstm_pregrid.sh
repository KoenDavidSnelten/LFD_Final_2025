#!/bin/bash
#SBATCH --job-name=lstm_grid
#SBATCH --output=/scratch/%u/LFD_Final_2025/lstm/output/other/slurm_logs/slurm-%A_%a.out
#SBATCH --partition=regular
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-23


echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# ------------------------------
# Load environment
# ------------------------------
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
source $HOME/venvs/venv_lfd_final/bin/activate

# ------------------------------
# Define preprocessing steps
# ------------------------------
PRE_STEPS=("lower_case" "no_emoji" "remove_hastag" "remove_html" "remove_url" "no_emoji_remove_contractions" "remove_contractions" "remove_swear")

# ------------------------------
# Define parameter grid (3 configs)
# ------------------------------
LRS=(0.003 0.003 0.003)
BATCH_SIZES=(16 16 64)
EPOCHS=(15 15 15)
LSTM_UNITS=(128 128 128)
DROPOUTS=(0.1 0.1 0.3)
LSTM_LAYERS=(2 2 2)
BIDIRECTIONAL_FLAGS=(True True True)
OPTIMIZERS=("Adam" "Adam" "Adam")

NUM_PARAMS=3
# CRITICAL FIX: Use ${#ARRAY[@]} to get the number of elements in a Bash array
NUM_PRE=${#PRE_STEPS[@]}
TOTAL=$(($NUM_PARAMS * $NUM_PRE)) # Should be 24

# ------------------------------
# Compute indices
# ------------------------------
idx=$SLURM_ARRAY_TASK_ID
PRE_IDX=$(($idx / $NUM_PARAMS))
PARAM_IDX=$(($idx % $NUM_PARAMS))

PRE_STEP=${PRE_STEPS[$PRE_IDX]}
LR=${LRS[$PARAM_IDX]}
BATCH_SIZE=${BATCH_SIZES[$PARAM_IDX]}
EPOCH=${EPOCHS[$PARAM_IDX]}
LSTM_UNIT=${LSTM_UNITS[$PARAM_IDX]}
DROPOUT=${DROPOUTS[$PARAM_IDX]}
LAYER=${LSTM_LAYERS[$PARAM_IDX]}
BIDIR=${BIDIRECTIONAL_FLAGS[$PARAM_IDX]}
OPTIMIZER=${OPTIMIZERS[$PARAM_IDX]}

# ------------------------------
# File and output setup
# ------------------------------
DATA_DIR="/scratch/$USER/LFD_Final_2025/data/$PRE_STEP"
TRAIN_FILE="$DATA_DIR/train.tsv"
DEV_FILE="$DATA_DIR/dev.tsv"
TEST_FILE="$DATA_DIR/test.tsv"

BASE_OUTPUT_DIR="/scratch/$USER/LFD_Final_2025/lstm/output/other/$PRE_STEP"
OUTPUT_DIR="$BASE_OUTPUT_DIR/lr_${LR}__bs_${BATCH_SIZE}__ep_${EPOCH}__units_${LSTM_UNIT}__drop_${DROPOUT}__layers_${LAYER}__bidir_${BIDIR}__opt_${OPTIMIZER}"
mkdir -p "$OUTPUT_DIR"

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

# ------------------------------
# Run the model
# ------------------------------
python3 lstm_model.py \
  --train_file "$TRAIN_FILE" \
  --dev_file "$DEV_FILE" \
  --test_file "$TEST_FILE" \
  --learning_rate "$LR" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$EPOCH" \
  --lstm_units "$LSTM_UNIT" \
  --dropout "$DROPOUT" \
  --lstm_layers "$LAYER" \
  --bidirectional_layer "$BIDIR" \
  --optimizer "$OPTIMIZER" \
  --verbose 0 \
  >> "$RESULTS_FILE" 2>&1


echo "Job finished. Results and model saved to $OUTPUT_DIR"

deactivate