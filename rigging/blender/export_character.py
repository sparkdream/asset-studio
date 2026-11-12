"""A rigged character for a game: its meshes, skeleton and sockets in their rest pose, no animation.

blender -b rigged.blend -P rigging/blender/export_character.py -- OUT.glb

Its clips are exported apart from the same rig (external/anymove/blender/apply_motion.py --glb --anim-only), so they
play on it by bone name (three.js: an AnimationMixer on this GLB's scene). The sockets (add_sockets.py) are nodes an
item is added under.
"""
import sys

import bpy

out = sys.argv[sys.argv.index("--") + 1]
rig = bpy.data.objects["rig"]
rig.animation_data_clear()
for pb in rig.pose.bones:
    pb.matrix_basis.identity()
meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() == rig]
bpy.ops.object.select_all(action="DESELECT")
for o in [rig, *meshes]:
    o.hide_set(False)
    o.select_set(True)
bpy.ops.export_scene.gltf(filepath=out, use_selection=True, export_animations=False, export_image_format="WEBP")
print(f"character: {len(meshes)} meshes, {sum(len(o.data.polygons) for o in meshes)} faces, "
      f"{len([b for b in rig.data.bones if b.name.startswith('socket_')])} sockets -> {out}")
