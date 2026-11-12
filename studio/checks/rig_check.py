"""Rig check: limb bones whose skin weights reach far from the limb (an arm that drags a coat panel with it).

blender -b -P studio/checks/rig_check.py -- rigged.blend creature.json out.json out.png
Writes, per limb bone, the vertices it moves (weight > 0.2) that sit further from it than 10% of the creature's size,
the same distance anymove's build_rig.py now fades limb weights out over, and a picture: front and side, the stray
vertices red over the limbs in their paint colours.
"""
import json, pathlib, sys

import bpy
import numpy as np

blend, creature, out_json, out_png = sys.argv[sys.argv.index("--") + 1:][:4]
bpy.ops.wm.open_mainfile(filepath=blend)
body, rig = bpy.data.objects["body"], bpy.data.objects["rig"]
paint = json.load(open(creature)).get("paint", {})

bones = {b.name: b for b in rig.data.bones}
limb_of = {n: k for k, v in paint.items() for n in v["bones"] if n in bones}
for n, b in bones.items():  # and what hangs off a limb (a blade, a hoof)
    p = b.parent
    while p and n not in limb_of:
        if p.name in limb_of:
            limb_of[n] = limb_of[p.name]
        p = p.parent

M = np.array(body.matrix_world)
co = np.array([v.co[:] for v in body.data.vertices]) @ M[:3, :3].T + M[:3, 3]
size = float(np.ptp(co, axis=0).max())
R = np.array(rig.matrix_world)
group = {g.index: g.name for g in body.vertex_groups}
weights = {n: np.zeros(len(co)) for n in limb_of}
for v in body.data.vertices:
    for g in v.groups:
        n = group.get(g.group)
        if n in weights:
            weights[n][v.index] = g.weight


def seg_dist(p, a, b):
    t = np.clip(((p - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
    return np.linalg.norm(a + (b - a) * t[:, None] - p, axis=1)


report, stray_all = [], np.zeros(len(co), bool)
for n, w in weights.items():
    # its reach: the bone and the weightless bones hanging off it (a held prop's bone runs from the hand to the
    # prop's tip, and the prop is skinned to the hand), so a held blade is not counted as dragged
    chain, todo = [bones[n]], list(bones[n].children)
    while todo:
        c = todo.pop()
        if c.name not in group.values():
            chain.append(c)
            todo += list(c.children)
    d = np.min([seg_dist(co, (R @ np.append(b.head_local, 1))[:3], (R @ np.append(b.tail_local, 1))[:3]) for b in chain], 0)
    moved = w > 0.2
    far = moved & (d > 0.10 * size)
    if far.sum() >= 50:
        p = co[far]
        report.append({"bone": n, "limb": limb_of[n], "stray": int(far.sum()), "moved": int(moved.sum()),
                       "share": round(float(far.sum() / moved.sum()), 3), "centroid": p.mean(0).round(3).tolist(),
                       "z": [round(float(p[:, 2].min()), 3), round(float(p[:, 2].max()), 3)]})
        stray_all |= far
report.sort(key=lambda r: -r["stray"])

# picture: front (x, z) and side (y, z) point clouds, limbs in their colours, strays red
S, pad = 420, 16
img = np.full((S + 2 * pad, 2 * S + 3 * pad, 3), 0.08, np.float32)
lo, hi = co.min(0), co.max(0)
scale = (S - 1) / max(hi[2] - lo[2], hi[0] - lo[0], hi[1] - lo[1])


def splat(mask, colour):
    for k, x0 in ((0, pad), (1, 2 * pad + S)):
        u = (x0 + (co[mask, k] - (lo[k] + hi[k]) / 2) * scale + S / 2).astype(int)
        v = (pad + (co[mask, 2] - lo[2]) * scale).astype(int)  # Blender images start at the bottom row
        img[v.clip(0, img.shape[0] - 1), u.clip(0, img.shape[1] - 1)] = colour


splat(np.ones(len(co), bool), (0.45, 0.45, 0.47))
for n, w in weights.items():
    hexcol = paint[limb_of[n]]["color"].lstrip("#")
    splat(w > 0.5, tuple(int(hexcol[i:i + 2], 16) / 255 for i in (0, 2, 4)))
splat(stray_all, (1.0, 0.1, 0.1))
im = bpy.data.images.new("rig_check", img.shape[1], img.shape[0], alpha=True)
im.pixels.foreach_set(np.concatenate([img, np.ones(img.shape[:2] + (1,), np.float32)], -1).ravel())
im.filepath_raw, im.file_format = str(out_png), "PNG"
im.save()

json.dump({"ok": not report, "size_m": round(size, 3), "stray_bones": report,
           "summary": "no limb drags the body with it" if not report else
           ", ".join(f"{r['bone']} drags {r['stray']} vertices up to {round(r['z'][1] - r['z'][0], 2)} m tall" for r in report)},
          open(out_json, "w"), indent=1)
print("rig check:", "ok" if not report else f"{len(report)} bones with stray weights", "->", out_json)
