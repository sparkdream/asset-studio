"""Skin check on a capture's animation: per frame and bone, how much of the mesh the bone drives collapses or stretches,
and how much of it sinks into another body part; and whether the GLB's own weights let a limb drag the body.

blender -b -P studio/checks/skin_check.py -- animation.glb creature.json out.json
Times are in seconds, so they line up with the clip whatever frame rate the GLB was exported at.
"""
import json, sys

import bpy
import numpy as np

glb, creature, out = sys.argv[sys.argv.index("--") + 1:][:3]
paint = json.load(open(creature)).get("paint", {})
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
bones = [b.name for b in arm.data.bones]
bi = {n: i for i, n in enumerate(bones)}
parent = [bi[b.parent.name] if b.parent else -1 for b in arm.data.bones]
limb_of = {n: k for k, v in paint.items() for n in v["bones"] if n in bi}
for b in arm.data.bones:  # and what hangs off a limb
    p = b.parent
    while p and b.name not in limb_of:
        if p.name in limb_of:
            limb_of[b.name] = limb_of[p.name]
        p = p.parent

# rest geometry, weights, the dominant bone of each vertex, edges
rest, W, edges, off = [], [], [], 0
for o in meshes:
    me = o.data
    co = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", co)
    M = np.array(o.matrix_world)
    rest.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
    gname = {g.index: g.name for g in o.vertex_groups}
    w = np.zeros((len(me.vertices), len(bones)), np.float32)
    for v in me.vertices:
        for g in v.groups:
            if gname.get(g.group) in bi:
                w[v.index, bi[gname[g.group]]] = g.weight
    e = np.empty(len(me.edges) * 2, np.int64)
    me.edges.foreach_get("vertices", e)
    W.append(w); edges.append(e.reshape(-1, 2) + off); off += len(me.vertices)
rest, W, edges = np.concatenate(rest), np.concatenate(W), np.concatenate(edges)
dom = np.where(W.max(1) > 0, W.argmax(1), -1)
rest_len = np.linalg.norm(rest[edges[:, 0]] - rest[edges[:, 1]], axis=1) + 1e-9
edge_bone = dom[edges[:, 0]]
size = float(np.ptp(rest, axis=0).max())
AM = np.array(arm.matrix_world)


def seg_dist(p, a, b):
    t = np.clip(((p - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
    return np.linalg.norm(a + (b - a) * t[:, None] - p, axis=1)


def chain_dist(a, b):
    up, x, n = {}, a, 0
    while x != -1:
        up[x] = n; x, n = parent[x], n + 1
    x, n = b, 0
    while x != -1 and x not in up:
        x, n = parent[x], n + 1
    return n + up[x] if x != -1 else 99


rest_seg = [((AM @ np.append(b.head_local, 1))[:3], (AM @ np.append(b.tail_local, 1))[:3]) for b in arm.data.bones]

# the GLB's own weights: a limb bone moving vertices far from it (same rule as rig_check.py)
strays = []
for n in limb_of:
    i = bi[n]
    far = (W[:, i] > 0.2) & (seg_dist(rest, *rest_seg[i]) > 0.10 * size)
    if far.sum() >= 50:
        strays.append({"bone": n, "limb": limb_of[n], "stray": int(far.sum())})

# each bone's thickness from its inner vertices: hair weighted to the head or a coat to the hips would make the
# median a poor measure of where the body really is
radius = np.zeros(len(bones))
for i in range(len(bones)):
    v = rest[dom == i]
    if len(v) > 20:
        radius[i] = np.percentile(seg_dist(v, *rest_seg[i]), 30)
pairs = [(a, b) for a in range(len(bones)) for b in range(len(bones))
         if a != b and radius[b] > 0 and (dom == a).sum() >= 20 and chain_dist(a, b) > 2]

scene = bpy.context.scene
action = arm.animation_data.action if arm.animation_data else None
f0, f1 = (int(action.frame_range[0]), int(action.frame_range[1])) if action else (scene.frame_start, scene.frame_end)
fps = scene.render.fps / scene.render.fps_base
frames = []
for f in range(f0, f1 + 1):
    scene.frame_set(f)
    dg = bpy.context.evaluated_depsgraph_get()
    cos = []
    for o in meshes:
        oe = o.evaluated_get(dg)
        me = oe.to_mesh()
        co = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", co)
        M = np.array(oe.matrix_world)
        cos.append(co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3])
        oe.to_mesh_clear()
    co = np.concatenate(cos)
    ratio = np.linalg.norm(co[edges[:, 0]] - co[edges[:, 1]], axis=1) / rest_len
    segs = [((AM @ np.append(np.array(pb.head), 1))[:3], (AM @ np.append(np.array(pb.tail), 1))[:3]) for pb in arm.pose.bones]
    rec = {"t": round((f - f0) / fps, 4), "squash": {}, "stretch": {}, "inside": []}
    for i, n in enumerate(bones):
        r = ratio[edge_bone == i]
        if len(r) > 50:
            if (r < 0.5).mean() > 0.002:
                rec["squash"][n] = round(float((r < 0.5).mean()), 4)
            if (r > 2.0).mean() > 0.002:
                rec["stretch"][n] = round(float((r > 2.0).mean()), 4)
    for a, b in pairs:
        frac = float((seg_dist(co[dom == a], *segs[b]) < 0.6 * radius[b]).mean())
        if frac > 0.05:
            rec["inside"].append([bones[a], bones[b], round(frac, 3)])
    frames.append(rec)
json.dump({"fps": fps, "bones": bones, "limb_of": limb_of, "strays": strays, "frames": frames}, open(out, "w"))
print("skin check:", len(frames), "frames,", len(strays), "bones with stray weights ->", out)
