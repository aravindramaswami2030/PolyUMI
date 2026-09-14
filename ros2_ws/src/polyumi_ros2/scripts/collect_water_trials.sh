#!/usr/bin/env bash
# Collect labelled water-bottle shake trials: record a bag while the arm does one fixed shake.
#
#   ./collect_water_trials.sh <level> <first_trial> [count] [duration_s]
#   ./collect_water_trials.sh empty 1 20        # 20 trials, empty bottle
#   ./collect_water_trials.sh half  1 20
#
# WHY A WRAPPER. The classifier's only premise is that the water level is the sole thing that
# differs between trials, so the recording window and the motion must line up the same way every
# time. Doing that by hand across 60 trials is how a label gets attached to a bag that caught half
# a shake. Here the recorder is started first, given time to subscribe, and stopped by its own
# duration; the shake is fired once in between.
#
# THE LABEL IS THE PATH. Bags land at $EVAL_ROOT/water/<level>/trial_<n>, reusing
# record_eval_trial.sh's <policy> slot as the class label -- so the training set is just a
# directory listing, with no separate manifest to drift out of sync.
#
# BETWEEN LEVELS, refill the bottle and start a new level; trial numbers are per level, so each
# run of this script can start again at 1. The shake returns the arm to where it started, so
# trials within a level need no reset.
set -euo pipefail

LEVEL="${1:?usage: collect_water_trials.sh <class-label> <first_trial> [count] [duration_s]}"
FIRST="${2:?missing first trial number}"
COUNT="${3:-1}"

# Shake geometry. These MUST be identical across every trial in a dataset -- the classifier's
# whole premise is that water level is the only thing that differs -- so they are env vars set
# once for a whole collection run rather than per-trial arguments, and they are echoed below so
# the value used is visible in the session log next to the bags it produced.
AMPLITUDE_M="${AMPLITUDE_M:-0.08}"
PERIOD_S="${PERIOD_S:-1.0}"
N_SHAKES="${N_SHAKES:-5}"
WAYPOINT_DT="${WAYPOINT_DT:-0.05}"
SHAKE_AXIS="${SHAKE_AXIS:-[0.0, 0.0, 1.0]}"
SYMMETRIC="${SYMMETRIC:-false}"

# Seconds the recorder is given to subscribe before the arm moves. Inside the recording window,
# so the derived duration below has to include it.
SUBSCRIBE_S=3

# Default the recording window to the motion it has to contain, rather than a fixed number that
# silently truncates a slower shake: the subscribe wait, 2 s levelling, 0.5 s settle, the shakes
# themselves, and 2 s margin for planning and for the arm to come to rest.
if [ -z "${4:-}" ]; then
    DURATION=$(awk -v s="${SUBSCRIBE_S}" -v n="${N_SHAKES}" -v p="${PERIOD_S}" \
               'BEGIN{printf "%d", s + 2 + 0.5 + n*p + 2 + 0.999}')
else
    DURATION="${4}"
fi

# The label is free text -- it names a class, and the same machinery serves water levels or
# materials or anything else. Only the character set is checked, because the label becomes a
# directory name and a stray slash or space would scatter a level across two places.
case "${LEVEL}" in
    *[!a-zA-Z0-9_-]*|'')
        echo "error: class label must be non-empty and [a-zA-Z0-9_-] only (got '${LEVEL}')" >&2
        exit 1 ;;
esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORDER="${HERE}/record_eval_trial.sh"
SHAKE_ARGS=(--ros-args -p execute:=true
            -p "amplitude_m:=${AMPLITUDE_M}"
            -p "period_s:=${PERIOD_S}"
            -p "n_shakes:=${N_SHAKES}"
            -p "waypoint_dt:=${WAYPOINT_DT}"
            -p "shake_axis:=${SHAKE_AXIS}"
            -p "symmetric:=${SYMMETRIC}")

echo "==> ${COUNT} trial(s) at level '${LEVEL}', starting at ${FIRST}, ${DURATION}s each"
echo "    shake: ${N_SHAKES} x ${PERIOD_S}s at ${AMPLITUDE_M}m along ${SHAKE_AXIS} (symmetric=${SYMMETRIC})"
echo "    bags -> ${EVAL_ROOT:-/data/eval_results_9_6}/water/${LEVEL}/"
echo

for i in $(seq 0 $((COUNT - 1))); do
    TRIAL=$((FIRST + i))
    echo "--- trial ${TRIAL} ($((i + 1))/${COUNT}) ---"

    # Recorder first, in the background, so it is subscribed before the arm moves. Its own
    # `timeout --signal=INT` ends it; we just wait for the process.
    TASK=water "${RECORDER}" "${LEVEL}" "${TRIAL}" "${DURATION}" &
    REC_PID=$!

    # ros2 bag needs a moment to discover publishers and subscribe. Starting the shake before it
    # has is the failure this sleep exists to prevent -- the bag would open, the arm would move,
    # and the first shake would be missing from the data with nothing to show it.
    sleep "${SUBSCRIBE_S}"

    ros2 run polyumi_ros2 water_shake "${SHAKE_ARGS[@]}" || {
        echo "WARNING: shake failed on trial ${TRIAL}; the bag will be short." >&2
    }

    # The recorder stops itself at DURATION. Waiting on it (rather than sleeping) keeps the loop
    # in step with the artefact, and surfaces its non-usable-trial check per trial rather than at
    # the end of a 20-trial run.
    if wait "${REC_PID}"; then
        echo "    ok"
    else
        echo "ERROR: trial ${TRIAL} was not usable -- see the recorder's message above." >&2
        echo "       Fix before continuing; the remaining trials would fail the same way." >&2
        exit 1
    fi
    echo
done

echo "==> done. ${COUNT} trial(s) at level '${LEVEL}'."
echo "    count them:  ls -d ${EVAL_ROOT:-/data/eval_results_9_6}/water/${LEVEL}/trial_* | wc -l"
