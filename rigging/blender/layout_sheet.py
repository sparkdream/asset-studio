"""A mesh's face layout as a picture: the whole body front and back, the head front and side, edges drawn over it.

blender -b --factory-startup -P rigging/blender/layout_sheet.py -- MODEL.obj|.glb OUT.png

An OBJ keeps quads as quads (rigging/optimise.py writes one for a quad layout); a GLB shows its triangles. The model
is taken to face +z in its own (y up) coordinates, as TRELLIS's GLBs do: -y in Blender's.
"""
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

src, out = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
if src.endswith(".obj"):
    bpy.ops.wm.obj_import(filepath=src)
else:
    bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])
h = hi.z - lo.z

solid = bpy.data.materials.new("solid")
solid.diffuse_color = (0.78, 0.78, 0.8, 1)
ink = bpy.data.materials.new("ink")
ink.diffuse_color = (0.05, 0.07, 0.1, 1)
for o in meshes:
    o.data.materials.clear()
    o.data.materials.append(solid)
    w = o.copy()
    w.data = o.data.copy()
    bpy.context.scene.collection.objects.link(w)
    m = w.modifiers.new("edges", "WIREFRAME")
    m.thickness, m.use_replace, m.use_even_offset = 0.0011 * h, True, False
    w.data.materials.clear()
    w.data.materials.append(ink)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "MATERIAL"
scene.render.resolution_x, scene.render.resolution_y = 700, 900
scene.world = bpy.data.worlds.new("w")
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"
centre = (lo + hi) / 2
head = Vector((centre.x, centre.y, hi.z - 0.075 * h))
tiles = []
for i, (target, direction, scale) in enumerate(((centre, (0, -1, 0), 1.08 * h), (centre, (0, 1, 0), 1.08 * h),
                                                (head, (0, -1, 0), 0.2 * h), (head, (1, 0, 0), 0.2 * h))):
    d = Vector(direction)
    cam.data.ortho_scale = scale
    cam.location = target + d * 3 * h
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{out}.{i}.png"
    bpy.ops.render.render(write_still=True)
    tiles.append(bpy.data.images.load(scene.render.filepath))

# the four side by side
W, H = scene.render.resolution_x, scene.render.resolution_y
rows = []
for t in tiles:
    a = np.empty(W * H * 4, np.float32)
    t.pixels.foreach_get(a)
    rows.append(a.reshape(H, W, 4))
sheet = bpy.data.images.new("sheet", W * 4, H)
sheet.pixels.foreach_set(np.concatenate(rows, axis=1).ravel())
sheet.filepath_raw, sheet.file_format = out, "PNG"
sheet.save()
for i in range(len(tiles)):
    os.remove(f"{out}.{i}.png")  # the tiles this script rendered
print(f"layout sheet -> {out}")
