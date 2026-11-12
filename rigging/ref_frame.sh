#!/bin/bash
# The two-camera reference frame a Wan clip starts from, rendered from the project's current rig with its limbs painted.
# rigging/ref_frame.sh PROJECT [NAME]
#   PROJECT: e.g. outputs/sparkdream/rogue_apose (anymove/creature.json, rigging/rigged/rigged.blend)
#   NAME: the frame's folder under anymove/runs (default ref): anymove/runs/NAME/ref.png
# rigging/promote.py runs it after the rig; run it again after rebuilding the rig.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
p=$(cd "$1" && pwd); name=${2:-ref}
am=$p/anymove; rig=$p/rigging/rigged; out=$am/runs/$name
py=${PYTHON:-$HOME/venvs/trellis2-xpu/bin/python}
mkdir -p "$out"
cd "$root"
export MOCAP_CREATURE=$am PYTHONPATH=$root/external/anymove
blender -b -P rigging/blender/paint_limbs.py -- "$rig/rigged.blend" "$am/creature.json" "$out/painted.blend" \
  | grep -E "limb paint|Error" || true
# rig_info.json beside the painted copy, for anything that opens it as a rig
cp "$rig/rig_info.json" "$out/rig_info.json"
blender -b -P external/anymove/blender/render_ref.py -- "$out/painted.blend" "$out" 64 | grep -E "Error" || true
"$py" external/anymove/tools/compose_ref.py "$out"
rm -f "$out/painted.blend" "$out/painted.blend1"
