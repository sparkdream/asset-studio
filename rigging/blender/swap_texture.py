"""Swap a packed image for another in the open blend (rigging/retexture.py): every image packed with exactly the bytes of
OLD gets NEW's, and the blend is saved (no .blend1 backup).

    blender -b --factory-startup FILE.blend -P swap_texture.py -- OLD.webp NEW.webp
"""
import sys

import bpy

old_path, new_path = sys.argv[sys.argv.index("--") + 1:]
old, new = open(old_path, "rb").read(), open(new_path, "rb").read()
hit = []
for im in bpy.data.images:
    if im.packed_file and bytes(im.packed_file.data) == old:
        im.pack(data=new, data_len=len(new))
        im.reload()
        hit.append(im.name)
if hit:
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_mainfile()
print(f"swap_texture: {', '.join(hit) + ' swapped' if hit else 'no image wore the old texture'}")
