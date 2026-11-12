"""Render the rigged character in its rest pose from the front and turned either way, for SAM 3D Body to read.

blender -b rigged.blend -P rigging/blender/render_rest.py -- OUT_DIR [--yaws -60,-30,0,30,60]

Writes OUT_DIR/rest_<i>.png and OUT_DIR/rest_cams.json: per image, its camera's rotation (camera axes to world:
x right, y up, z toward the camera, as SAM's rotations are given), and its framing (the point it centres on, its
orthographic height in metres, its resolution). Orthographic, like the clips' front view, on the same plain backdrop.
"""
import json, math, pathlib, sys

import bpy
from mathutils import Euler, Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
out = pathlib.Path(args[0])
out.mkdir(parents=True, exist_ok=True)
yaws = [float(y) for y in args[args.index("--yaws") + 1].split(",")] if "--yaws" in args else [-60.0, -30.0, 0.0, 30.0, 60.0]

scene = bpy.context.scene
meshes = [o for o in scene.objects if o.type == "MESH" and o.visible_get()]
pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
lo = Vector([min(p[i] for p in pts) for i in range(3)])
hi = Vector([max(p[i] for p in pts) for i in range(3)])
centre, height = (lo + hi) / 2, hi.z - lo.z

scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "TEXTURE"
scene.render.resolution_x, scene.render.resolution_y = 540, 720
scene.render.film_transparent = False
if scene.world is None:
    scene.world = bpy.data.worlds.new("world")
scene.world.color = (0.216, 0.223, 0.235)  # the clips' backdrop, (128, 130, 134) in sRGB
scene.display_settings.display_device = "sRGB"
scene.view_settings.view_transform = "Standard"

cam = bpy.data.objects.new("rest_cam", bpy.data.cameras.new("rest_cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"
cam.data.sensor_fit = "VERTICAL"
cam.data.ortho_scale = height * 1.15
cams = []
for i, yaw in enumerate(yaws):
    rot = Matrix.Rotation(math.radians(yaw), 3, "Z") @ Euler((math.pi / 2, 0, 0)).to_matrix()
    cam.matrix_world = Matrix.Translation(centre + rot @ Vector((0, 0, 10))) @ rot.to_4x4()
    scene.render.filepath = str(out / f"rest_{i}.png")
    bpy.ops.render.render(write_still=True)
    cams.append({"image": f"rest_{i}.png", "yaw": yaw, "rot": [list(r) for r in rot], "centre": list(centre),
                 "ortho_scale": cam.data.ortho_scale, "res": [scene.render.resolution_x, scene.render.resolution_y]})
json.dump({"cams": cams}, open(out / "rest_cams.json", "w"), indent=1)
print("rest renders:", len(cams), "->", out)
