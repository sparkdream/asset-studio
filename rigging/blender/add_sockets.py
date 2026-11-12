"""Sockets: where items attach to a rig. Bones that deform nothing, fixed to their parent, that a game (or the studio's
preview) attaches an item's grip to: an item's origin goes on the socket's head, its axes on the socket's.

blender -b rigged.blend -P rigging/blender/add_sockets.py -- rig_info.json CHARACTER.json [--check OTHER_rig_info.json]

Run after the rig is built (rigging/rig.sh); running it again replaces the sockets. The skin weights are untouched,
and captures stay valid: a socket has no keys of its own, so it follows its hand (export a capture again to have the
sockets in its GLB).

socket_hand_L, socket_hand_R, from the hand's own vertices (those mostly weighted to it):
- the head: the middle of the fist, the centroid of the hand's vertices past the first third of the hand bone (the
  wrist and the back of the hand left out), which is where a handle passes;
- the axes: the hand's direction d (wrist to knuckles), the fist's hole k (the hand's widest direction across d: where
  a handle runs in a hammer grip; signed to the thumb, which points forward in the template's rest pose, palms in) and
  the back of the hand n = d x k (signed away from the body). The socket's Y, where an item's tip points, is
  grip_deg off the hand towards the thumb (90: a hammer grip, square to the forearm; the default 45, as
  build_skeleton.py re-grips a held prop, because no wrist bends far enough to lay a blade along the arm); its Z, where
  an item's face looks, is the back of the hand.
CHARACTER.json "sockets": {"grip_deg": 45} overrides it, or {"grip_deg": {"L": 57, "R": 70}} per hand: the thumb's
direction k comes from each fist's own shape, so two hands can grip a few tens of degrees apart (the rogue's daggers,
at 45 each, stood upright in the right hand and leant forward in the left).

The item convention (studio items, item.json): origin at the middle of the grip, +Y towards the tip, +Z out of the
face (a blade's flat, a shield's front).

--check: compare with another rig of the same character whose weapon bones held a real prop (each weapon bone runs
from the hand along the blade): how far the socket's head lies from the blade's line, and the angle between them.
"""
import json, sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
info_path, char_path = args[:2]
check = args[args.index("--check") + 1] if "--check" in args else None
info = json.load(open(info_path))
char = json.load(open(char_path))
grip_cfg = char.get("sockets", {}).get("grip_deg", 45)
grip_of = lambda s: float(grip_cfg.get(s, 45) if isinstance(grip_cfg, dict) else grip_cfg)

rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
co = np.array([body.matrix_world @ v.co for v in body.data.vertices])
gi = {g.name: g.index for g in body.vertex_groups}


def weight(group):
    w = np.zeros(len(co))
    g = gi.get(group)
    if g is None:
        return w
    for v in body.data.vertices:
        for e in v.groups:
            if e.group == g:
                w[v.index] = e.weight
    return w


sockets = {}
for s in ("L", "R"):
    hand = f"hand_{s}"
    b = rig.data.bones[hand]
    w, t = np.array(rig.matrix_world @ b.head_local), np.array(rig.matrix_world @ b.tail_local)
    d = (t - w) / np.linalg.norm(t - w)
    pts = co[weight(hand) > 0.5]
    if len(pts) < 20:
        print(f"{hand}: only {len(pts)} vertices weighted to it: no socket")
        continue
    along = (pts - w) @ d / np.linalg.norm(t - w)
    c = pts[along > 1 / 3].mean(0)
    q = pts - pts.mean(0)
    q -= np.outer(q @ d, d)  # across the hand
    k = np.linalg.svd(q, full_matrices=False)[2][0]
    k -= (k @ d) * d
    k /= np.linalg.norm(k)
    if k[1] > 0:  # the thumb points forward (-y)
        k = -k
    n = np.cross(d, k)
    if n[0] * (1 if s == "L" else -1) < 0:  # the back of the hand faces away from the body (left hand: +x)
        n = -n
    grip_deg = grip_of(s)
    a = np.radians(grip_deg)
    y = np.cos(a) * d + np.sin(a) * k
    z = n - (n @ y) * y
    z /= np.linalg.norm(z)
    sockets[f"socket_hand_{s}"] = {"parent": hand, "head": c, "y": y, "z": z}
    print(f"socket_hand_{s}: head {np.round(c, 3).tolist()}, Y {np.round(y, 2).tolist()}, Z {np.round(z, 2).tolist()} "
          f"({len(pts)} hand vertices, grip {grip_deg:g} deg)")

# the bones, replacing any made before
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
eb = rig.data.edit_bones
for e in [e for e in eb if e.name.startswith("socket_")]:
    eb.remove(e)
inv = rig.matrix_world.inverted()
for name, sk in sockets.items():
    e = eb.new(name)
    e.head = inv @ Vector(sk["head"])
    e.tail = inv @ Vector(sk["head"] + 0.08 * sk["y"])
    e.align_roll(inv.to_3x3() @ Vector(sk["z"]))
    e.parent = eb[sk["parent"]]
    e.use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")
for name in sockets:
    rig.data.bones[name].use_deform = False
bpy.context.preferences.filepaths.save_version = 0  # no rigged.blend1 beside it
bpy.ops.wm.save_mainfile()

names = [b["name"] for b in info["bones"] if not b["name"].startswith("socket_")]
info["bones"] = [b for b in info["bones"] if not b["name"].startswith("socket_")]
for name, sk in sockets.items():
    bb = rig.data.bones[name]
    info["bones"].append({"name": name, "parent": names.index(sk["parent"]), "matrix_local": [list(r) for r in bb.matrix_local],
                          "length": bb.length, "head_joint": f"{name}.head", "tail_joint": f"{name}.tail",
                          "limits_deg": [[0, 0]] * 3})
    names.append(name)
info["sockets"] = {name: {"parent": sk["parent"], "grip_deg": grip_deg} for name, sk in sockets.items()}
json.dump(info, open(info_path, "w"), indent=1)
print(f"sockets: {', '.join(sockets)} -> {info_path}")

if check:
    other = {b["name"]: b for b in json.load(open(check))["bones"]}
    for s in ("L", "R"):
        wb, sk = other.get(f"weapon_{s}"), sockets.get(f"socket_hand_{s}")
        if not wb or not sk:
            continue
        M = np.array(wb["matrix_local"])
        a, u = M[:3, 3], M[:3, 1]
        off = sk["head"] - a
        dist = np.linalg.norm(off - (off @ u) * u)
        ang = np.degrees(np.arccos(np.clip(u @ sk["y"], -1, 1)))
        print(f"check {s}: the socket's head {dist * 100:.1f} cm from the other rig's blade line, its Y {ang:.0f} deg "
              f"from the blade")
