#!/bin/bash
# Build a character's rig from its unrigged mesh: rest renders -> SAM 3D Body -> joints -> skeleton, weights -> hair.
# rigging/rig.sh CHARACTER_DIR
#   CHARACTER_DIR/character.json: "static" (the unrigged .blend, its mesh named "body", relative to the folder),
#   "template" (rigging/templates/<name>.json), "props", "hair", "layout", "camera".
# Writes CHARACTER_DIR/fit/ (SAM's reading of the bare mesh, mhr_rest.json) and CHARACTER_DIR/rigged/.
# The hair still goes through anymove's add_hair.py until it is ported (docs/rigging-plan.md).
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
c=$(cd "$1" && pwd)
sam_py=${SAM_PYTHON:-$HOME/venvs/sam3d-xpu/bin/python}
read -r static template hair < <(python3 -c "
import json, os; c = json.load(open('$c/character.json'))
print(os.path.normpath(os.path.join('$c', c['static'])), c.get('template', 'humanoid'), int('hair' in c))")
mkdir -p "$c/fit"
cd "$root"
blender -b "$static" -P rigging/blender/render_rest.py -- "$c/fit" | grep -E "rest renders|Error" || true
"$sam_py" -m rigging.sam_body "$c/fit/rest_sam.npz" "$c"/fit/rest_[0-9].png 2>&1 | grep -E "^saved|Error" || true
python3 -m rigging.fit "$c/fit" "$c/fit/mhr_rest.json"
blender -b "$static" -P rigging/blender/build_skeleton.py -- "rigging/templates/$template.json" "$c/fit/mhr_rest.json" \
  "$c/character.json" "$c/rigged" | grep -E "centred|axis|grip|cut|without weights|makehuman|spread|split|fixed in|saved|check|Error" || true
if [ "$hair" = 1 ]; then
  # the hair (or fur, mane...) as SAM 3 sees it in renders all the way round, whatever its colour; character.json
  # "hair": {"prompt": "fur"} names it otherwise. Without SAM 3, add_hair falls back to its colour rule.
  prompt=$(python3 -c "import json; print(json.load(open('$c/character.json'))['hair'].get('prompt', 'hair'))")
  r=$c/rigged/region
  rm -f "$c/rigged/hair_region.npy"
  blender -b "$c/rigged/rigged.blend" -P rigging/blender/region_views.py -- render "$r" | grep -E "region views|Error" || true
  if "$sam_py" -m rigging.segment "$prompt" "$r/hair" "$r"/view_[0-9].png 2>&1 | grep -E "instance|Error"; then
    blender -b "$c/rigged/rigged.blend" -P rigging/blender/region_views.py -- project "$r" hair "$c/rigged/hair_region.npy" \
      | grep -E "region|Error" || true
  fi
  (cd external/anymove && blender -b -P blender/add_hair.py -- "$c/rigged/rigged.blend" "$c/character.json" \
    | grep -E "^hair|strands|split|Error" || true)
fi
# loose cloth below the hips (coat panels, skirts) on spring bones of its own, not dragged by the legs
blender -b -P rigging/blender/add_cloth.py -- "$c/rigged/rigged.blend" "$c/character.json" | grep -E "^cloth|Error" || true
# loose pieces lying on the surface (straps, trims, a slit's edges) move with it, cloth included
blender -b "$c/rigged/rigged.blend" -P rigging/blender/follow_surface.py | grep -E "^follow|Error" || true
# sockets: where items attach (the hands), bones that deform nothing
blender -b "$c/rigged/rigged.blend" -P rigging/blender/add_sockets.py -- "$c/rigged/rig_info.json" "$c/character.json" \
  | grep -E "^socket|Error" || true
echo "rig: $c/rigged (check the joints: $c/rigged/joints_front.png, joints_side.png)"
