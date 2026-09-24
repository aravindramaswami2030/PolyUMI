#!/usr/bin/env bash
# Collect the whole texture dataset: 50 finger-camera samples for each of 5 classes, interleaved.
#
#   ./collect_texture_dataset.sh                          # dry run: nothing moves, nothing saved
#   EXECUTE=true ./collect_texture_dataset.sh             # the real thing, 250 samples
#   EXECUTE=true SAMPLES=25 ./collect_texture_dataset.sh  # a short rehearsal first
#   CONFIG=~/my_textures.json EXECUTE=true ./collect_texture_dataset.sh
#
# WHAT IT DOES per sample: moves the arm to a fresh random pose, then waits for three ENTERs --
# open the gripper, close it, capture the image. The object to fetch is named for you; it is
# drawn from the class whose block is running, so the operator never picks it.
#
# THE OBJECT LIST IS YOURS. Copy config/texture_objects.example.json, fill in your five classes
# and the objects in each, and point CONFIG at it. Several objects per class is the point: a class
# collected from a single object is a dataset about that object, and nothing in the images would
# reveal the difference at training time.
#
# INTERLEAVED IN BLOCKS, for the same reason collect_water_dataset.sh is: the finger camera
# settles at a slightly different level and colour balance each time the rig is disturbed, and
# collected one class at a time that offset IS the label -- a classifier scored 93% on exactly
# that nuisance in the water corpus. BLOCK samples of each class in turn removes the alignment.
#
# ONE GRIP WIDTH for every object and every class, calibrated on the first grip and then reused
# from ${ROOT}/grip_width_m. A width measured per object would make the grip a perfect predictor
# of the class. The same goes for the pose reference in ${ROOT}/home_xyz: poses are jittered about
# one recorded point rather than about wherever the arm happens to be, so the start cannot
# random-walk across the session.
#
# RESUMING is automatic: each class's next sample number is counted off disk, so a session that
# stops part way continues where it left off when you re-run this.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-${EVAL_ROOT:-/data}/texture_dataset}"
CONFIG="${CONFIG:-${HERE}/../config/texture_objects.example.json}"
SAMPLES="${SAMPLES:-50}"
BLOCK="${BLOCK:-10}"
EXECUTE="${EXECUTE:-false}"
JITTER_XYZ_M="${JITTER_XYZ_M:-[0.04, 0.04, 0.03]}"
YAW_DEG="${YAW_DEG:-25.0}"
TILT_DEG="${TILT_DEG:-8.0}"
MOVE_TIME_S="${MOVE_TIME_S:-3.0}"
GRIP_SQUEEZE_M="${GRIP_SQUEEZE_M:-0.002}"
SEED="${SEED:--1}"

GRIP_FILE="${ROOT}/grip_width_m"
HOME_FILE="${ROOT}/home_xyz"
GRIP_WIDTH_M="${GRIP_WIDTH_M:-}"
HOME_XYZ="${HOME_XYZ:-}"
if [ -z "${GRIP_WIDTH_M}" ] && [ -r "${GRIP_FILE}" ]; then
    GRIP_WIDTH_M="$(cat "${GRIP_FILE}")"
    echo "==> grip width ${GRIP_WIDTH_M}m read from ${GRIP_FILE}"
fi
if [ -z "${HOME_XYZ}" ] && [ -r "${HOME_FILE}" ]; then
    HOME_XYZ="$(cat "${HOME_FILE}")"
    echo "==> pose reference ${HOME_XYZ} read from ${HOME_FILE}"
fi

if [ ! -r "${CONFIG}" ]; then
    echo "error: no object config at ${CONFIG}" >&2
    echo "       copy ${HERE}/../config/texture_objects.example.json, edit it, and set CONFIG=" >&2
    exit 1
fi
CLASS_COUNT="$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len([k for k in d if not k.startswith('_')]))" "${CONFIG}")"

echo "=============================================================="
echo " texture dataset: ${SAMPLES} samples x ${CLASS_COUNT} classes = $((SAMPLES * CLASS_COUNT)) images"
echo " interleaved in blocks of ${BLOCK}; objects from ${CONFIG}"
echo " pose: reference ${HOME_XYZ:-<measured now>} +/- ${JITTER_XYZ_M} m, yaw +/-${YAW_DEG}deg, tilt +/-${TILT_DEG}deg"
echo " grip: ${GRIP_WIDTH_M:-<calibrated on the first grip>} m, the same for every class"
echo " images -> ${ROOT}/<class>/<class>_<object>_<n>.png   (+ manifest.jsonl)"
echo " execute: ${EXECUTE}"
echo "=============================================================="
echo
echo "Before starting, confirm ALL of these:"
echo "  * the NUC is up with execute_arm:=true, and policy_client_node is NOT running"
echo "  * the Pi is streaming the finger camera (polyumi-pi stream)"
echo "  * the arm is somewhere roomy, and you can reach the jaws to swap objects"
echo "  * you have dry-run this once (EXECUTE unset) and watched /polyumi/target_poses_preview"
echo
read -r -p "Ready? [Enter to begin, Ctrl-C to abort] " _

ARGS=(--ros-args
      -p "config:=${CONFIG}"
      -p "root:=${ROOT}"
      -p "samples_per_class:=${SAMPLES}"
      -p "block:=${BLOCK}"
      -p "seed:=${SEED}"
      -p "jitter_xyz_m:=${JITTER_XYZ_M}"
      -p "yaw_deg:=${YAW_DEG}"
      -p "tilt_deg:=${TILT_DEG}"
      -p "move_time_s:=${MOVE_TIME_S}"
      -p "grip_squeeze_m:=${GRIP_SQUEEZE_M}"
      -p "execute:=${EXECUTE}")
[ -n "${GRIP_WIDTH_M}" ] && ARGS+=(-p "grip_width_m:=${GRIP_WIDTH_M}")
[ -n "${HOME_XYZ}" ] && ARGS+=(-p "home_xyz:=${HOME_XYZ}")

mkdir -p "${ROOT}"
# Not a bare mktemp: an exported TMPDIR naming a directory that does not exist makes it fail, and
# under `set -e` that would abort the run before a single sample.
mkdir -p "${TMPDIR:-/tmp}" 2>/dev/null || true
LOG="$(mktemp 2>/dev/null || mktemp -p /tmp)"
ros2 run polyumi_ros2 texture_collect "${ARGS[@]}" 2>&1 | tee "${LOG}"

# Capture the calibrated grip width and pose reference the first session reports, so every later
# session -- and every later class -- reuses them rather than measuring again.
if [ -z "${GRIP_WIDTH_M}" ]; then
    CAPTURED="$(sed -n 's/.*GRIP_TARGET_M=\([0-9.]*\).*/\1/p' "${LOG}" | head -1)"
    if [ -n "${CAPTURED}" ]; then
        printf '%s\n' "${CAPTURED}" > "${GRIP_FILE}"
        echo "==> calibrated grip ${CAPTURED}m -> ${GRIP_FILE} (every later sample reuses it)"
    fi
fi
if [ -z "${HOME_XYZ}" ]; then
    CAPTURED="$(sed -n 's/.*HOME_XYZ=\([0-9eE.,+-]*\).*/\1/p' "${LOG}" | head -1)"
    if [ -n "${CAPTURED}" ]; then
        printf '[%s]\n' "${CAPTURED}" > "${HOME_FILE}"
        echo "==> pose reference [${CAPTURED}] -> ${HOME_FILE}"
    fi
fi
rm -f "${LOG}"

echo
echo "=============================================================="
echo " per-class counts:"
for DIR in "${ROOT}"/*/; do
    [ -d "${DIR}" ] || continue
    printf '   %-18s %s\n' "$(basename "${DIR}")" "$(find "${DIR}" -name '*.png' | wc -l)"
done
echo
echo " the tree is torchvision ImageFolder shaped, so a simple CNN trains straight off it:"
echo "   datasets.ImageFolder('${ROOT}', transform=...)"
echo " manifest.jsonl carries the object, pose and grip per sample -- use it to check whether a"
echo " model is separating textures or just memorising individual objects."
echo "=============================================================="
