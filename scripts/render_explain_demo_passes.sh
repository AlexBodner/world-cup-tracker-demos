#!/usr/bin/env bash
# Render all pass-explain assets on pass-network demo detections (v11 @ ball_thr=0.20).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.

VIDEO=world_cup_projects/bundesliga_videos/08fd33_0.mp4
OUT_ROOT=world_cup_projects/assets/explain_frames
COMMON=(
  --detector-backend yolo
  --ball-threshold 0.20
  --metric --layout talk --explain-video --gif
)

passes=( "8:10" "10:20" "20:14" "3:27" "27:3" "3:16" "16:6" )

for spec in "${passes[@]}"; do
  passer=${spec%%:*}
  receiver=${spec##*:}
  echo "========== #${passer} → #${receiver} =========="
  python -m world_cup_projects.explain.pass_explain_run \
    --video "$VIDEO" \
    "${COMMON[@]}" \
    --passer-tid "$passer" --receiver-tid "$receiver" \
    --out "${OUT_ROOT}/pass_${passer}_to_${receiver}"
done

echo "Done."
