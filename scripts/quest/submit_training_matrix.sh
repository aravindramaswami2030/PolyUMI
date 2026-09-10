#!/usr/bin/env bash
# Build a compatible dataset x model matrix and submit it as one Slurm array.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_training_matrix.sh --datasets FILE --models FILE [options]

Options:
  --max-parallel N   Maximum simultaneous GPU jobs (default: 2)
  --account NAME     Slurm account (default: p52914)
  --partition NAME   Slurm partition (default: gengpu)
  --gpu RESOURCE     Slurm GRES request (default: gpu:1; any GPU model)
  --time HH:MM:SS    Per-task time limit (default: 02:00:00)
  --cpus N           CPUs per task (default: 8)
  --mem SIZE         Memory per task (default: 64G)
  --wandb-mode MODE  offline, online, or disabled (default: offline)
  --wandb-entity ID  W&B team/entity (default: cwhayes)
  --wandb-project ID W&B project (default: polyumi-quest)
  --dry-run          Validate and print the generated matrix without submitting
EOF
}

DATASETS_FILE=""
MODELS_FILE=""
MAX_PARALLEL=2
ACCOUNT=p52914
PARTITION=gengpu
GPU_RESOURCE=gpu:1
TIME_LIMIT=02:00:00
CPUS=8
MEMORY=64G
WANDB_MODE=offline
WANDB_ENTITY="${WANDB_ENTITY:-cwhayes}"
WANDB_PROJECT="${WANDB_PROJECT:-polyumi-quest}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets) DATASETS_FILE="$2"; shift 2 ;;
        --models) MODELS_FILE="$2"; shift 2 ;;
        --max-parallel) MAX_PARALLEL="$2"; shift 2 ;;
        --account) ACCOUNT="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --gpu) GPU_RESOURCE="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --mem) MEMORY="$2"; shift 2 ;;
        --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
        --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$DATASETS_FILE" && -n "$MODELS_FILE" ]] || { usage >&2; exit 2; }
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "max-parallel must be positive" >&2; exit 2; }
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || { echo "cpus must be positive" >&2; exit 2; }
case "$WANDB_MODE" in offline|online|disabled) ;; *) echo "invalid W&B mode" >&2; exit 2 ;; esac

DATASETS_FILE="$(realpath -e "$DATASETS_FILE")"
MODELS_FILE="$(realpath -e "$MODELS_FILE")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_ROOT="${POLYUMI_TRAIN_ROOT:-$(dirname "$REPO_ROOT")}"
GENERATED_DIR="$TRAIN_ROOT/manifests/generated"
mkdir -p "$GENERATED_DIR" "$TRAIN_ROOT/logs"

declare -a DATASET_NAMES DATASET_PATHS DATASET_TYPES
while IFS=$'\t' read -r NAME PATH_VALUE TYPE REST; do
    NAME="${NAME%$'\r'}"; PATH_VALUE="${PATH_VALUE%$'\r'}"; TYPE="${TYPE%$'\r'}"
    [[ -z "$NAME" || "$NAME" == \#* || "$NAME" == name ]] && continue
    [[ -z "$REST" ]] || { echo "too many dataset columns for $NAME" >&2; exit 2; }
    [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid dataset name: $NAME" >&2; exit 2; }
    [[ "$PATH_VALUE" = /* ]] || { echo "dataset path must be absolute: $PATH_VALUE" >&2; exit 2; }
    [[ -f "$PATH_VALUE" ]] || { echo "dataset not found: $PATH_VALUE" >&2; exit 2; }
    case "$TYPE" in dp|polyumi) ;; *) echo "dataset type must be dp or polyumi: $NAME" >&2; exit 2 ;; esac
    DATASET_NAMES+=("$NAME"); DATASET_PATHS+=("$PATH_VALUE"); DATASET_TYPES+=("$TYPE")
done < "$DATASETS_FILE"

declare -a MODEL_NAMES MODEL_POLICIES MODEL_VARIANTS MODEL_EPOCHS MODEL_BATCHES
while IFS=$'\t' read -r NAME POLICY VARIANT EPOCHS BATCH REST; do
    NAME="${NAME%$'\r'}"; POLICY="${POLICY%$'\r'}"; VARIANT="${VARIANT%$'\r'}"
    EPOCHS="${EPOCHS%$'\r'}"; BATCH="${BATCH%$'\r'}"
    [[ -z "$NAME" || "$NAME" == \#* || "$NAME" == name ]] && continue
    [[ -z "$REST" ]] || { echo "too many model columns for $NAME" >&2; exit 2; }
    [[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid model name: $NAME" >&2; exit 2; }
    [[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "invalid epochs for $NAME" >&2; exit 2; }
    [[ "$BATCH" =~ ^[1-9][0-9]*$ ]] || { echo "invalid batch size for $NAME" >&2; exit 2; }
    case "$POLICY" in
        dp)
            [[ -f "$REPO_ROOT/external/polyumi_diffusion_policy/diffusion_policy/config/${VARIANT}.yaml" ]] || {
                echo "unknown DP config for $NAME: $VARIANT" >&2; exit 2;
            }
            ;;
        vista)
            case "$VARIANT" in
                polytouch|see_hear_feel|sparsh_x|qformer|mitas|vta_diffusion|\
                qformer_vt|qformer_va|qformer_v|mitas_vt|mitas_va|mitas_v|\
                vista|vista_vt|vista_va|vista_v) ;;
                *) echo "unknown Vista model for $NAME: $VARIANT" >&2; exit 2 ;;
            esac
            ;;
        *) echo "policy must be dp or vista: $NAME" >&2; exit 2 ;;
    esac
    MODEL_NAMES+=("$NAME"); MODEL_POLICIES+=("$POLICY"); MODEL_VARIANTS+=("$VARIANT")
    MODEL_EPOCHS+=("$EPOCHS"); MODEL_BATCHES+=("$BATCH")
done < "$MODELS_FILE"

(( ${#DATASET_NAMES[@]} > 0 )) || { echo "no datasets found" >&2; exit 2; }
(( ${#MODEL_NAMES[@]} > 0 )) || { echo "no models found" >&2; exit 2; }

MATRIX_FILE="$GENERATED_DIR/matrix-$(date +%Y%m%d-%H%M%S)-$$.tsv"
printf 'dataset_name\tdataset_path\tdataset_type\tmodel_name\tpolicy\tvariant\tepochs\tbatch_size\n' > "$MATRIX_FILE"
COUNT=0
for ((D=0; D<${#DATASET_NAMES[@]}; D++)); do
    for ((M=0; M<${#MODEL_NAMES[@]}; M++)); do
        # DP can ignore extra PolyUMI streams; Vista requires the multimodal export.
        [[ "${MODEL_POLICIES[M]}" == vista && "${DATASET_TYPES[D]}" != polyumi ]] && continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${DATASET_NAMES[D]}" "${DATASET_PATHS[D]}" "${DATASET_TYPES[D]}" \
            "${MODEL_NAMES[M]}" "${MODEL_POLICIES[M]}" "${MODEL_VARIANTS[M]}" \
            "${MODEL_EPOCHS[M]}" "${MODEL_BATCHES[M]}" >> "$MATRIX_FILE"
        ((++COUNT))
    done
done
(( COUNT > 0 )) || { echo "no compatible dataset/model pairs" >&2; exit 2; }

echo "Generated $COUNT tasks in $MATRIX_FILE"
column -t -s $'\t' "$MATRIX_FILE" 2>/dev/null || sed -n '1,20p' "$MATRIX_FILE"
if (( DRY_RUN )); then
    exit 0
fi

export POLYUMI_TRAIN_ROOT="$TRAIN_ROOT"
export POLYUMI_REPO_ROOT="$REPO_ROOT"
export WANDB_MODE
export WANDB_ENTITY
export WANDB_PROJECT
JOB_ID="$(sbatch --parsable \
    --account="$ACCOUNT" --partition="$PARTITION" --gres="$GPU_RESOURCE" \
    --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEMORY" --time="$TIME_LIMIT" \
    --array="1-${COUNT}%${MAX_PARALLEL}" --job-name=polyumi-matrix \
    --output="$TRAIN_ROOT/logs/%x-%A_%a.out" \
    "$SCRIPT_DIR/train_matrix_task.sbatch" "$MATRIX_FILE" "$REPO_ROOT")"
echo "Submitted array job $JOB_ID"
echo "Monitor with: squeue -j $JOB_ID"
