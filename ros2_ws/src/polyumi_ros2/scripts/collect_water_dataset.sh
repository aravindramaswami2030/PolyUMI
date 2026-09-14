#!/usr/bin/env bash
# Collect the whole water-level dataset: 30 trials at each of three levels, pausing to refill.
#
#   ./collect_water_dataset.sh              # 30 trials x empty, some, half
#   ./collect_water_dataset.sh 30 half some empty
#   TRIALS=5 ./collect_water_dataset.sh     # short rehearsal before committing to the full set
#   ./collect_water_dataset.sh 25 plastic black_metal silver_metal   # any class labels
#
# DURATION_S overrides the derived recording window. Set it whenever the shake is CLAMPED by the
# controller -- a request over max_pos_speed is stretched, not truncated, so the motion runs
# longer than n_shakes*period_s and the derived window would cut it off mid-trial.
#   AMPLITUDE_M=0.12 PERIOD_S=0.5 ./collect_water_dataset.sh   # a faster, larger shake
#
# SHAKE GEOMETRY is set by AMPLITUDE_M (default 0.08 m), PERIOD_S (default 1.0 s) and N_SHAKES
# (default 5), and passed through to every trial. Change them for a whole dataset, never between
# levels: the classifier assumes water level is the only thing that differs, so a dataset shaken
# two ways has a second, unlabelled variable in it. The recording window is derived from them, so
# a slower shake is not silently truncated.
#
# Speed scales as AMPLITUDE_M/PERIOD_S and acceleration as AMPLITUDE_M/PERIOD_S^2, so shortening
# the period bites much harder than raising the amplitude. At the defaults the peak is 0.25 m/s
# and 1.6 m/s^2, about a quarter of the controller's own max_pos_speed and ~5% of its 20 N force
# ceiling against a 0.7 kg payload -- there is a lot of headroom for a more vigorous shake.
#
# Runs collect_water_trials.sh once per level and STOPS between levels so the bottle can be
# refilled -- it will not start the next level until you confirm, because an unattended run that
# rolled straight on would silently label 30 trials with the previous level's water.
#
# RESUMING. Trial numbers are per level and the recorder refuses to overwrite, so a run that dies
# partway is resumed by starting the level again with a higher first trial:
#     ./collect_water_trials.sh some 18 13
#
# The last 5 trials of each level become the validation split (water_dataset.py --val-from 26),
# so collect them in the same session and the same way as the first 25 -- they are the held-out
# test of whether the classifier generalises, and a change of setup between 25 and 26 would show
# up as a generalisation failure that is really a procedure change.
set -euo pipefail

TRIALS="${TRIALS:-${1:-30}}"
shift || true
AMPLITUDE_M="${AMPLITUDE_M:-0.08}"
PERIOD_S="${PERIOD_S:-1.0}"
N_SHAKES="${N_SHAKES:-5}"
WAYPOINT_DT="${WAYPOINT_DT:-0.05}"
SHAKE_AXIS="${SHAKE_AXIS:-[0.0, 0.0, 1.0]}"
SYMMETRIC="${SYMMETRIC:-false}"
DURATION_S="${DURATION_S:-}"
export AMPLITUDE_M PERIOD_S N_SHAKES
LEVELS=("$@")
# half -> some -> empty: you can always POUR WATER OUT between levels, but refilling to an
# exact level mid-run is fiddly and drifts. Starting full and emptying keeps the bottle, the
# grip and the setup identical down the sequence, so the only thing changing is the water.
[ ${#LEVELS[@]} -eq 0 ] && LEVELS=(half some empty)
export WAYPOINT_DT SHAKE_AXIS SYMMETRIC

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PER_LEVEL="${HERE}/collect_water_trials.sh"
ROOT="${EVAL_ROOT:-/data/eval_results_9_6}/water"

echo "=============================================================="
echo " water dataset: ${TRIALS} trials x ${#LEVELS[@]} levels (${LEVELS[*]})"
echo " shake: ${N_SHAKES} x ${PERIOD_S}s at ${AMPLITUDE_M}m  (identical for every trial)"
echo " bags -> ${ROOT}/<level>/trial_<n>"
echo " the last 5 trials of each level are the validation split"
echo "=============================================================="
echo
echo "Before starting, confirm ALL of these -- a wrong one costs the whole level:"
echo "  * the NUC is up with execute_arm:=true"
echo "  * the Pi is streaming (tactile + audio) and the GoPro is up"
echo "  * the bottle is gripped, and the arm is somewhere roomy"
echo "  * you have dry-run the motion once:  ros2 run polyumi_ros2 water_shake"
echo
read -r -p "Ready? [Enter to begin, Ctrl-C to abort] " _

for i in "${!LEVELS[@]}"; do
    LEVEL="${LEVELS[$i]}"
    echo
    echo "=============================================================="
    echo " LEVEL $((i + 1))/${#LEVELS[@]}: ${LEVEL}"
    echo "=============================================================="
    echo
    echo "  >>> Set the bottle to '${LEVEL}'."
    echo "      Pour water OUT to reach it -- do not re-grip unless you have to, since a"
    echo "      changed grip is a second variable the classifier cannot tell from water."
    echo
    read -r -p "  Press ENTER when the water level is set and the bottle is held... " _

    START=1
    if compgen -G "${ROOT}/${LEVEL}/trial_*" > /dev/null; then
        # Continue past whatever is already there rather than failing on the first collision --
        # a half-finished level is the normal way this script gets interrupted.
        LAST=$(basename "$(ls -d "${ROOT}/${LEVEL}"/trial_* | sed 's/.*trial_//' | sort -n | tail -1)")
        START=$((LAST + 1))
        echo "  note: ${LEVEL} already has trials up to ${LAST}; continuing from ${START}"
    fi
    REMAINING=$((TRIALS - START + 1))
    if [ "${REMAINING}" -le 0 ]; then
        echo "  ${LEVEL} already has ${TRIALS} trials -- skipping."
        continue
    fi

    "${PER_LEVEL}" "${LEVEL}" "${START}" "${REMAINING}" ${DURATION_S}
    echo "  ${LEVEL}: $(ls -d "${ROOT}/${LEVEL}"/trial_* 2>/dev/null | wc -l) trial(s) on disk"
done

echo
echo "=============================================================="
echo " done. per-level counts:"
for LEVEL in "${LEVELS[@]}"; do
    printf '   %-8s %s\n' "${LEVEL}" "$(ls -d "${ROOT}/${LEVEL}"/trial_* 2>/dev/null | wc -l)"
done
echo
echo " next, offline:"
echo "   python3 analysis/water_dataset.py --root ${ROOT} --out water_tensors.npz"
echo "   python3 analysis/train_water_cnn.py --tensors water_tensors.npz"
echo "=============================================================="
