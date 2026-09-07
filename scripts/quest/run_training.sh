#!/usr/bin/env bash
# Run one PolyUMI training experiment inside an allocated Quest GPU job.
set -euo pipefail

usage() {
    echo "usage: $0 POLICY DATASET VARIANT EPOCHS BATCH_SIZE DATASET_NAME MODEL_NAME" >&2
    exit 2
}

[[ $# -eq 7 ]] || usage

POLICY="$1"
DATASET="$2"
VARIANT="$3"
EPOCHS="$4"
BATCH_SIZE="$5"
DATASET_NAME="$6"
MODEL_NAME="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_ROOT="${POLYUMI_TRAIN_ROOT:-$(dirname "$REPO_ROOT")}"

[[ "$DATASET" = /* ]] || { echo "dataset path must be absolute: $DATASET" >&2; exit 2; }
[[ -f "$DATASET" ]] || { echo "dataset not found: $DATASET" >&2; exit 2; }
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "epochs must be a positive integer" >&2; exit 2; }
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "batch size must be a positive integer" >&2; exit 2; }
[[ "$DATASET_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid dataset name: $DATASET_NAME" >&2; exit 2; }
[[ "$MODEL_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid model name: $MODEL_NAME" >&2; exit 2; }

case "$POLICY" in
    dp)
        POLICY_DIR="$REPO_ROOT/external/polyumi_diffusion_policy"
        [[ -f "$POLICY_DIR/diffusion_policy/config/${VARIANT}.yaml" ]] || {
            echo "unknown DP config: $VARIANT" >&2
            exit 2
        }
        ;;
    vista)
        POLICY_DIR="$REPO_ROOT/external/polyumi_vista_policy"
        case "$VARIANT" in
            polytouch|see_hear_feel|sparsh_x|vista) ;;
            *) echo "unknown Vista model: $VARIANT" >&2; exit 2 ;;
        esac
        ;;
    *) echo "policy must be dp or vista" >&2; exit 2 ;;
esac

PYTHON="$TRAIN_ROOT/envs/$POLICY/bin/python"
[[ -x "$PYTHON" ]] || { echo "environment is missing: $TRAIN_ROOT/envs/$POLICY" >&2; exit 2; }

RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}"
STARTED_AT="$(date --iso-8601=seconds)"
OUTPUT_DIR="${OUTPUT_DIR:-$TRAIN_ROOT/outputs/$DATASET_NAME/$MODEL_NAME/$RUN_ID}"
mkdir -p "$OUTPUT_DIR" "$TRAIN_ROOT/cache/huggingface" "$TRAIN_ROOT/cache/torch" \
    "$TRAIN_ROOT/cache/numba" "$TRAIN_ROOT/cache/matplotlib"

export PATH="$TRAIN_ROOT/envs/$POLICY/bin:$PATH"
export HF_HOME="$TRAIN_ROOT/cache/huggingface"
export TORCH_HOME="$TRAIN_ROOT/cache/torch"
export NUMBA_CACHE_DIR="$TRAIN_ROOT/cache/numba"
export MPLCONFIGDIR="$TRAIN_ROOT/cache/matplotlib"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
WANDB_MODE="${WANDB_MODE:-offline}"

cd "$POLICY_DIR"
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0))'

printf 'policy=%s\ndataset=%s\nvariant=%s\nepochs=%s\nbatch_size=%s\njob_id=%s\nstarted_at=%s\n' \
    "$POLICY" "$DATASET" "$VARIANT" "$EPOCHS" "$BATCH_SIZE" "$RUN_ID" "$STARTED_AT" \
    > "$OUTPUT_DIR/run-metadata.txt"

if [[ "$POLICY" == dp ]]; then
    "$PYTHON" train.py \
        --config-name="$VARIANT" \
        "task.dataset_path=$DATASET" \
        "hydra.run.dir=$OUTPUT_DIR" \
        "training.num_epochs=$EPOCHS" \
        training.checkpoint_every=5 \
        "dataloader.batch_size=$BATCH_SIZE" \
        "val_dataloader.batch_size=$BATCH_SIZE" \
        "dataloader.num_workers=${DATALOADER_WORKERS:-4}" \
        "val_dataloader.num_workers=${DATALOADER_WORKERS:-4}" \
        "logging.mode=$WANDB_MODE"
else
    export DAY0SUITE_DATASET="$DATASET"
    bash scripts/train_day0suite.sh \
        --model "$VARIANT" \
        "hydra.run.dir=$OUTPUT_DIR" \
        "training.num_epochs=$EPOCHS" \
        training.checkpoint_every=5 \
        "dataloader.batch_size=$BATCH_SIZE" \
        "val_dataloader.batch_size=$BATCH_SIZE" \
        "dataloader.num_workers=${DATALOADER_WORKERS:-4}" \
        "val_dataloader.num_workers=${DATALOADER_WORKERS:-4}" \
        "logging.mode=$WANDB_MODE"
fi

touch "$OUTPUT_DIR/SUCCESS"
