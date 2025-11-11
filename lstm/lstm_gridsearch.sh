#!/bin/bash
#SBATCH --job-name=lstm_grid
#SBATCH --output=/scratch/%u/LFD_Final_2025/lstm/output/raw/slurm_logs/slurm-%A_%a.out
#SBATCH --partition=regular
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-863  # 2*3*1*2*3*2*2*2 = 864 total jobs

echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

# Load environment
module load CUDA/11.7.0
module load Boost/1.79.0-GCC-11.3.0
source $HOME/venvs/venv_lfd_final/bin/activate

# Default file locations
PRE_STEP="raw"
DATA_DIR="/scratch/$USER/LFD_Final_2025/data/$PRE_STEP"
TRAIN_FILE="$DATA_DIR/train.tsv"
DEV_FILE="$DATA_DIR/dev.tsv"
TEST_FILE="$DATA_DIR/test.tsv"

BASE_OUTPUT_DIR="/scratch/$USER/LFD_Final_2025/lstm/output/$PRE_STEP"
mkdir -p $BASE_OUTPUT_DIR

# ------------------------------
# Define the hyperparameter grid
# ------------------------------
LRS=(0.001 0.003)
BATCH_SIZES=(16 32 64)
EPOCHS=(15)
LSTM_UNITS=(64 128)
DROPOUTS=(0.1 0.2 0.3)
LSTM_LAYERS=(1 2)
BIDIRECTIONAL_FLAGS=(True False)
OPTIMIZERS=("SGD" "Adam")

# Grid sizes
NUM_LRS=${#LRS[@]}
NUM_BATCH=${#BATCH_SIZES[@]}
NUM_EPOCHS=${#EPOCHS[@]}
NUM_UNITS=${#LSTM_UNITS[@]}
NUM_DROPOUTS=${#DROPOUTS[@]}
NUM_LAYERS=${#LSTM_LAYERS[@]}
NUM_BIDIRS=${#BIDIRECTIONAL_FLAGS[@]}
NUM_OPTS=${#OPTIMIZERS[@]}

# Total combinations = 2*3*1*2*3*2*2*2 = 864
# ------------------------------
# Map SLURM array ID to parameters
# ------------------------------
idx=$SLURM_ARRAY_TASK_ID

OPT_IDX=$((idx % NUM_OPTS))
BIDIR_IDX=$(((idx / NUM_OPTS) % NUM_BIDIRS))
LAYER_IDX=$(((idx / (NUM_OPTS * NUM_BIDIRS)) % NUM_LAYERS))
DROP_IDX=$(((idx / (NUM_OPTS * NUM_BIDIRS * NUM_LAYERS)) % NUM_DROPOUTS))
UNIT_IDX=$(((idx / (NUM_OPTS * NUM_BIDIRS * NUM_LAYERS * NUM_DROPOUTS)) % NUM_UNITS))
EPOCH_IDX=$(((idx / (NUM_OPTS * NUM_BIDIRS * NUM_LAYERS * NUM_DROPOUTS * NUM_UNITS)) % NUM_EPOCHS))
BATCH_IDX=$(((idx / (NUM_OPTS * NUM_BIDIRS * NUM_LAYERS * NUM_DROPOUTS * NUM_UNITS * NUM_EPOCHS)) % NUM_BATCH))
LR_IDX=$(((idx / (NUM_OPTS * NUM_BIDIRS * NUM_LAYERS * NUM_DROPOUTS * NUM_UNITS * NUM_EPOCHS * NUM_BATCH)) % NUM_LRS))

# Get parameter values
LR=${LRS[$LR_IDX]}
BATCH_SIZE=${BATCH_SIZES[$BATCH_IDX]}
EPOCH=${EPOCHS[$EPOCH_IDX]}
LSTM_UNIT=${LSTM_UNITS[$UNIT_IDX]}
DROPOUT=${DROPOUTS[$DROP_IDX]}
LAYER=${LSTM_LAYERS[$LAYER_IDX]}
BIDIR=${BIDIRECTIONAL_FLAGS[$BIDIR_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}

# Output directory
OUTPUT_DIR="$BASE_OUTPUT_DIR/lr_${LR}__bs_${BATCH_SIZE}__ep_${EPOCH}__units_${LSTM_UNIT}__drop_${DROPOUT}__layers_${LAYER}__bidir_${BIDIR}__opt_${OPTIMIZER}"
mkdir -p $OUTPUT_DIR
RESULTS_FILE="$OUTPUT_DIR/results.txt"

# Write run info
{
  echo "--- Grid Search Run ---"
  echo "Job ID: $SLURM_ARRAY_TASK_ID"
  echo "Learning Rate: $LR"
  echo "Batch Size: $BATCH_SIZE"
  echo "Epochs: $EPOCH"
  echo "LSTM Units: $LSTM_UNIT"
  echo "Dropout: $DROPOUT"
  echo "LSTM Layers: $LAYER"
  echo "Bidirectional: $BIDIR"
  echo "Optimizer: $OPTIMIZER"
  echo "Output Dir: $OUTPUT_DIR"
  echo "-----------------------"
  echo ""
  echo "--- Training Log ---"
} > $RESULTS_FILE

# Run LSTM training
python lstm_model.py \
    --train_file $TRAIN_FILE \
    --dev_file $DEV_FILE \
    --test_file $TEST_FILE \
    --learning_rate $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCH \
    --lstm_units $LSTM_UNIT \
    --dropout $DROPOUT \
    --lstm_layers $LAYER \
    --bidirectional $BIDIR \
    --optimizer $OPTIMIZER \
    --verbose 0 \
    >> $RESULTS_FILE 2>&1

echo "Job finished. Results and model saved to $OUTPUT_DIR"

deactivate