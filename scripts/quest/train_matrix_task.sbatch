#!/bin/bash
set -euo pipefail

MATRIX_FILE="${1:?Generated matrix file is required}"
REPO_ROOT="${2:?Repository root is required}"
[[ -f "$MATRIX_FILE" ]] || { echo "matrix file not found: $MATRIX_FILE" >&2; exit 2; }
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "run this script as a Slurm array" >&2; exit 2; }

ROW="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MATRIX_FILE")"
[[ -n "$ROW" ]] || { echo "no matrix row for task $SLURM_ARRAY_TASK_ID" >&2; exit 2; }

IFS=$'\t' read -r DATASET_NAME DATASET DATASET_TYPE MODEL_NAME POLICY VARIANT EPOCHS BATCH_SIZE <<< "$ROW"
echo "matrix task: dataset=$DATASET_NAME ($DATASET_TYPE), model=$MODEL_NAME, policy=$POLICY"

exec "$REPO_ROOT/scripts/quest/run_training.sh" \
    "$POLICY" "$DATASET" "$VARIANT" "$EPOCHS" "$BATCH_SIZE" \
    "$DATASET_NAME" "$MODEL_NAME"
