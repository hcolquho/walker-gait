#!/bin/bash
#SBATCH --job-name=walker-gait-process
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/process_%A_%a.out
#SBATCH --error=logs/process_%A_%a.err
#
# Runs scripts/process_session.py on every trial folder under
# data/raw_video/han/, one trial per SLURM array task, in parallel.
#
# Submit with an array size matching the trial count, e.g.:
#   N=$(python scripts/process_all_sessions.sh --count)
#   sbatch --array=0-$((N - 1)) scripts/process_all_sessions.sh
#
# Assumes batch_inference.py has already been run for each trial, with
# its output JSON mirrored under RESULTS_ROOT using the same relative
# path as the trial folder (see KEYPOINTS_JSON below).

RAW_VIDEO_ROOT="data/raw_video/han"
RESULTS_ROOT="results/han"

# Camera mount calibration -- fixed for the rig unless remounted.
TILT_DEG="${TILT_DEG:-35}"
FLOOR_OFFSET_MM="${FLOOR_OFFSET_MM:-950}"
FPS="${FPS:-30}"

mkdir -p logs

# Build the trial list: every directory containing timing_marks_manual.json.
mapfile -t TRIALS < <(find "$RAW_VIDEO_ROOT" -type f -name timing_marks_manual.json \
    -exec dirname {} \; | sort)

if [[ "$1" == "--count" ]]; then
    echo "${#TRIALS[@]}"
    exit 0
fi

if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    echo "This script must be submitted via sbatch --array=0-N-1 (see header comment)." >&2
    exit 1
fi

TRIAL_DIR="${TRIALS[$SLURM_ARRAY_TASK_ID]}"
if [[ -z "$TRIAL_DIR" ]]; then
    echo "No trial for array index $SLURM_ARRAY_TASK_ID (found ${#TRIALS[@]} trials)." >&2
    exit 1
fi

RELATIVE_PATH="${TRIAL_DIR#"$RAW_VIDEO_ROOT"/}"
KEYPOINTS_JSON="$RESULTS_ROOT/$RELATIVE_PATH.json"

source activate walker-gait

echo "[$SLURM_ARRAY_TASK_ID] Processing $TRIAL_DIR"
python scripts/process_session.py \
    --session "$TRIAL_DIR" \
    --keypoints-json "$KEYPOINTS_JSON" \
    --tilt-deg "$TILT_DEG" \
    --floor-offset-mm "$FLOOR_OFFSET_MM" \
    --fps "$FPS"
