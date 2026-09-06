#!/bin/bash
#SBATCH --account=p52914
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=polyumi-one
#SBATCH --output=/projects/p52914/xph8283/polyumi/logs/%x-%j.out
set -euo pipefail

POLICY="${1:?Specify dp or vista}"
DATASET="${2:?Specify an absolute dataset path}"

if [[ "$POLICY" == dp ]]; then
    VARIANT="${3:-train_diffusion_unet_timm_polyumi_workspace}"
else
    VARIANT="${3:-vista_vt}"
fi

EPOCHS="${4:-2}"
BATCH_SIZE="${5:-8}"
DATASET_NAME="$(basename "$DATASET")"
DATASET_NAME="${DATASET_NAME%.zarr.zip}"
MODEL_NAME="${6:-$VARIANT}"

REPO_ROOT="${POLYUMI_REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[[ -x "$REPO_ROOT/scripts/quest/run_training.sh" ]] || {
    echo "Cannot locate the PolyUMI checkout. Submit from its root or set POLYUMI_REPO_ROOT." >&2
    exit 2
}
exec "$REPO_ROOT/scripts/quest/run_training.sh" \
    "$POLICY" "$DATASET" "$VARIANT" "$EPOCHS" "$BATCH_SIZE" \
    "$DATASET_NAME" "$MODEL_NAME"
