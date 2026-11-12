"""Find a region of the mesh (hair, fur, a mane, a cape...) by what a segmentation model sees in renders of it, not by
its colour: render the mesh all the way round, let rigging/segment.py mark the region in each render, then each
vertex takes the vote of the views that see it.

blender -b rigged.blend -P rigging/blender/region_views.py -- render OUT_DIR [--views 8] [--res 768]
    OUT_DIR/view_<i>.png (textured, orthographic, the camera going round at the mesh's mid-height) and cams.json
blender -b rigged.blend -P rigging/blender/region_views.py -- project OUT_DIR MASK_PREFIX OUT.npy
    reads OUT_DIR/<MASK_PREFIX>_<i>.png (white = the region) and writes, per vertex of the mesh "body" in this .blend,
    the share of the views that see it which put it in the region (vertices no view sees take their neighbours').
Vertex indices are this .blend's: project on the same file the result is used on.
"""
import json, math, sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index("--") + 1:]
mode, out = args[0], Path(args[1])
body = bpy.data.objects["body"]
scene = bpy.context.scene


def opt(name, default):
    return type(default)(args[args.index(name) + 1]) if name in args else default


def camera(cam_info):
    cam = bpy.data.objects.new("region_cam", bpy.data.cameras.new("region_cam"))
    scene.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = cam_info["ortho_scale"]
    cam.matrix_world = Matrix(cam_info["matrix"])
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = cam_info["res"]
    scene.render.resolution_percentage = 100
    return cam


if mode == "render":
    out.mkdir(parents=True, exist_ok=True)
    n, res = opt("--views", 8), opt("--res", 768)
    for o in scene.objects:  # the mesh alone, as it is at rest
        o.hide_render = o is not body
    co = np.array([body.matrix_world @ v.co for v in body.data.vertices])
    lo, hi = co.min(0), co.max(0)
    centre, size = (lo + hi) / 2, hi - lo
    scale = 1.08 * max(size[2], np.hypot(size[0], size[1]))
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "FLAT"  # the texture's own colours, as an image model expects to see them
    scene.display.shading.color_type = "TEXTURE"
    scene.render.film_transparent = False
    scene.world = scene.world or bpy.data.worlds.new("region_world")
    scene.world.color = (0.75, 0.75, 0.75)  # a plain light grey background
    cams = []
    for i in range(n):
        yaw = 2 * math.pi * i / n  # 0: in front (the mesh faces -Y), then round to its left
        d = Vector((math.sin(yaw), -math.cos(yaw), 0))
        loc = Vector(centre) + d * (size.max() + 2)
        rot = (-d).to_track_quat("-Z", "Y").to_matrix().to_4x4()
        info = {"image": f"view_{i}.png", "yaw_deg": math.degrees(yaw), "ortho_scale": float(scale), "res": [res, res],
                "matrix": [list(r) for r in (Matrix.Translation(loc) @ rot)]}
        camera(info)
        scene.render.filepath = str(out / info["image"])
        bpy.ops.render.render(write_still=True)
        cams.append(info)
    json.dump({"cams": cams}, open(out / "cams.json", "w"), indent=1)
    print(f"region views: {n} -> {out}")

elif mode == "project":
    prefix, dst = args[2], Path(args[3])
    cams = json.load(open(out / "cams.json"))["cams"]
    me = body.data
    nv = len(me.vertices)
    co = [body.matrix_world @ v.co for v in me.vertices]
    bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())
    votes, seen = np.zeros(nv), np.zeros(nv)
    for i, info in enumerate(cams):
        mask_path = out / f"{prefix}_{i}.png"
        img = bpy.data.images.load(str(mask_path))
        w, h = img.size
        mask = np.array(img.pixels[:]).reshape(h, w, img.channels)[..., 0] > 0.5  # rows bottom-up, as camera y
        cam = camera(info)
        bpy.context.view_layer.update()
        toward = Vector(cam.matrix_world.col[2][:3]).normalized()  # the camera looks down its -Z
        size = max(w, h) * 1e-4
        for v, p in enumerate(co):
            hit = bvh.ray_cast(p + toward * 1e-3, toward)
            if hit[0] is not None:  # something lies between this vertex and the camera
                continue
            uv = world_to_camera_view(scene, cam, p)
            x, y = int(uv.x * (w - 1) + 0.5), int(uv.y * (h - 1) + 0.5)
            if 0 <= x < w and 0 <= y < h:
                seen[v] += 1
                votes[v] += mask[y, x]
        bpy.data.objects.remove(cam)
        print(f"{mask_path.name}: {int(mask.sum())} region pixels", flush=True)
    share = np.where(seen > 0, votes / np.maximum(seen, 1), np.nan)
    E = np.array([e.vertices[:] for e in me.edges])
    for _ in range(200):  # the unseen (inside double shells, under the region) take their seen neighbours' share
        unk = np.isnan(share)
        if not unk.any():
            break
        val, cnt = np.zeros(nv), np.zeros(nv)
        for a, b in ((E[:, 0], E[:, 1]), (E[:, 1], E[:, 0])):
            ok = ~np.isnan(share[b])
            np.add.at(val, a[ok], share[b[ok]])
            np.add.at(cnt, a[ok], 1)
        fill = unk & (cnt > 0)
        share[fill] = val[fill] / cnt[fill]
    share = np.nan_to_num(share)
    np.save(dst, share.astype(np.float32))
    print(f"region: {int((share > 0.5).sum())} of {nv} vertices (seen by a view: {int((seen > 0).sum())}) -> {dst}")
else:
    sys.exit(f"unknown mode {mode}")
