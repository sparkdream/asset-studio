#!/bin/bash
# Capture a clip with SAM 3D Body retargeting: video -> SAM's joint rotations -> the rig -> render, GLB, showcase.
# rigging/capture.sh CHARACTER_DIR VIDEO.mp4 NAME
#   CHARACTER_DIR: e.g. outputs/sparkdream/rogue/rigging, with character.json and the rig from rigging/rig.sh in rigged/
# CLIPS=DIR: where clips/NAME goes (default CHARACTER_DIR/clips). SPEED=1.5: play faster.
# CALIB=DET.npz: a calibration from the video's pixels to the layout (anymove's det_reader.npz); else a plain resize.
# PROPS=DET.npz: a keypoint reading with the held props' points (anymove's det_reader.npz) for the prop touch-up.
# SHOWCASE_CREATURE=DIR: with CALIB, the anymove creature folder that reading came from, for the showcase's layout.
# Rendering and the showcase still run through anymove's scripts until they are ported (docs/rigging-plan.md).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
c=$(cd "$1" && pwd); video=$(realpath "$2"); name=$3
sam_py=${SAM_PYTHON:-$HOME/venvs/sam3d-xpu/bin/python}
py=${PYTHON:-$HOME/venvs/trellis2-xpu/bin/python}
rig=$c/rigged; out=$(realpath -m "${CLIPS:-$c/clips}/$name")
mkdir -p "$c/rest" "$out"
cd "$root"
read -r front cam target lens < <("$py" -c "
import json; c = json.load(open('$c/character.json')); k = c.get('camera', {})
print(c.get('layout', {}).get('W_FRONT', 420), k.get('cam', '-3.2,-3.0,1.6'), k.get('target', '0,0,0.5'), k.get('lens', 50))")
# SAM's reading of the character's rest pose, redone when the rig is rebuilt
if [ ! -f "$c/rest/rest_sam.npz" ] || [ "$rig/rigged.blend" -nt "$c/rest/rest_sam.npz" ]; then
  blender -b "$rig/rigged.blend" -P rigging/blender/render_rest.py -- "$c/rest" | grep -E "rest renders|Error" || true
  "$sam_py" -m rigging.sam_body "$c/rest/rest_sam.npz" "$c"/rest/rest_[0-9].png 2>&1 | grep -E "^saved|Error" || true
fi
"$sam_py" -m rigging.sam_body "$out/sam.npz" "$video" --view "0,$front" ${CALIB:+--calib "$CALIB"} 2>&1 | grep -E "^saved|Error" || true
blender -b "$rig/rigged.blend" -P rigging/blender/retarget.py -- "$rig/retarget_map.json" "$c/rest" "$out/sam.npz" \
  "$rig/rig_info.json" "$out/motion.json" --speed "${SPEED:-1}" ${PROPS:+--props "$PROPS"} \
  | grep -E "rest reading|turned|retargeted|Error" || true
cd external/anymove
blender -b -P blender/apply_motion.py -- "$rig/rigged.blend" "$rig/rig_info.json" "$out/motion.json" "$out/render" \
  --views persp,side --samples 24 --cam "$cam" --target "$target" --lens "$lens" --glb \
  $([ -f "$rig/springs.json" ] && echo --springs "$rig/springs.json") > "$out/blender.log" 2>&1
if [ -n "${CALIB:-}" ]; then  # the showcase draws the reader's points, so it needs a reading
  cp "$CALIB" "$out/det.npz" 2>/dev/null || true
  MOCAP_CREATURE=${SHOWCASE_CREATURE:?the anymove creature folder the reading came from} "$py" -m mocap.showcase "$video" "$out/det.npz" "$out/render" "$out/showcase.mp4" "$name" \
    && ffmpeg -loglevel error -y -i "$out/showcase.mp4" -c:v libx264 -crf 20 -pix_fmt yuv420p "$out/showcase_h264.mp4"
fi
echo "done: $out (animation: $out/render/animation.glb)"
