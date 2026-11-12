"""Reduce and unwrap a mesh for rigging/optimise.py: triangles (Decimate) or quads (QuadriFlow).

blender -b --factory-startup -P rigging/blender/optimise_mesh.py -- in.npz out.npz MODE TARGET_FACES ISLAND_MARGIN

in.npz: vertices (N, 3), faces (F, 3), the detailed surface (o_voxel's remesh_snap). MODE:
- tris: decimated to TARGET_FACES triangles, as postprocess_torch.to_glb does;
- quads (an experiment, from the command line only: the face comes out torn, docs/rigging-plan.md): QuadriFlow to
  TARGET_FACES / 2 quads (each quad is two triangles in a GLB), from a copy decimated to at most QUADRIFLOW_INPUT faces
  first (QuadriFlow is slow and memory-hungry on a million faces); QuadriFlow keeps its quads on
  that surface. OPTIMISE_SHRINKWRAP=NEAREST_SURFACEPOINT (or another Shrinkwrap wrap method) pulls them back onto the
  detailed surface too: an experiment, as nearest points jump between layers lying close (hair over a cheek).
Then Smart UV Project and packing (island_margin: a fraction of the texture).
out.npz: vertices (V, 3), loop_start, loop_total (P,), loop_vert, loop_uv (the UV of each face corner): the faces as
made, quads or triangles.
"""
import math
import os
import sys

import bpy
import numpy as np

QUADRIFLOW_INPUT = 300000
src, dst, mode, target, margin = sys.argv[sys.argv.index("--") + 1:]
target, margin = int(target), float(margin)
d = np.load(src)
bpy.ops.wm.read_factory_settings(use_empty=True)


def mesh_object(name, v, f):
    me = bpy.data.meshes.new(name)
    me.from_pydata(v.astype(np.float32), [], f.astype(np.int32))
    o = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(o)
    return o


def activate(o):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = o
    o.select_set(True)


def decimate(o, faces):
    if 0 < faces < len(o.data.polygons):
        m = o.modifiers.new("decimate", "DECIMATE")
        m.ratio = faces / len(o.data.polygons)
        m.use_collapse_triangulate = True
        activate(o)
        bpy.ops.object.modifier_apply(modifier=m.name)


hi = mesh_object("detailed", d["vertices"], d["faces"])
if mode == "tris":
    work = hi
    decimate(work, target)
elif mode == "quads":
    work = mesh_object("work", d["vertices"], d["faces"])
    decimate(work, QUADRIFLOW_INPUT)
    activate(work)
    n = len(work.data.polygons)
    res = bpy.ops.object.quadriflow_remesh(target_faces=max(target // 2, 100), use_preserve_sharp=False,
                                           use_preserve_boundary=False, use_mesh_symmetry=False, smooth_normals=False,
                                           seed=0)
    if res != {"FINISHED"} or len(work.data.polygons) == n:
        sys.exit(f"QuadriFlow failed ({res}): the surface may not be manifold")
    if os.environ.get("OPTIMISE_SHRINKWRAP"):  # experiment: pull the quads back onto the detailed surface
        sw = work.modifiers.new("onto the detailed surface", "SHRINKWRAP")
        sw.target, sw.wrap_method = hi, os.environ["OPTIMISE_SHRINKWRAP"]
        activate(work)
        bpy.ops.object.modifier_apply(modifier=sw.name)
    sizes = np.array([len(p.vertices) for p in work.data.polygons])
    print(f"quads: {int((sizes == 4).sum())} quads, {int((sizes != 4).sum())} other faces", flush=True)
else:
    sys.exit(f"unknown mode {mode}")

activate(work)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin, margin_method="FRACTION")
bpy.ops.uv.select_all(action="SELECT")
bpy.ops.uv.pack_islands(rotate=True, margin_method="FRACTION", margin=margin)
bpy.ops.object.mode_set(mode="OBJECT")

me = work.data
v = np.empty(len(me.vertices) * 3, np.float32)
me.vertices.foreach_get("co", v)
ls, lt = np.empty(len(me.polygons), np.int32), np.empty(len(me.polygons), np.int32)
me.polygons.foreach_get("loop_start", ls)
me.polygons.foreach_get("loop_total", lt)
lv = np.empty(len(me.loops), np.int32)
me.loops.foreach_get("vertex_index", lv)
uv = np.empty(len(me.loops) * 2, np.float32)
me.uv_layers.active.data.foreach_get("uv", uv)
np.savez(dst, vertices=v.reshape(-1, 3), loop_start=ls, loop_total=lt, loop_vert=lv, loop_uv=uv.reshape(-1, 2))
print(f"{mode}: {len(me.vertices)} vertices, {len(me.polygons)} faces", flush=True)
