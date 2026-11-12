"""Drive a rig from another rig's motion (both from the same template): each bone turns in the world as the other
rig's does, and each limb points where the other's points.

blender -b rigged.blend -P rigging/blender/retarget_rig.py -- MAP.json SRC_rig_info.json SRC_motion.json rig_info.json
    OUT_motion.json

MAP.json is the template's Kimodo map (rigging/templates/humanoid_kimodo.json); only its "aim" bones (the limbs) and
"feet" are read. A motion.json turns each bone from its own rig's rest pose, and each rig rests differently (each is
fitted to its mesh: one rig's upper arm rests 49 degrees below horizontal, another's 76), so the Euler angles can't be
copied. Instead, as retarget_kimodo.py does from Kimodo: each bone's world turn since its rest, D = P A^T (P its posed
orientation, A its rest, both in world axes), is applied to this rig's bone, then each limb is turned the least so it
points along the other rig's limb (which keeps the roll). Bones this rig lacks (a weapon bone) are left out; bones
only this rig has turn with their parent.

The hips travel as the other rig's do, scaled by the two rigs' hip heights; with the other rig's joint positions in
the motion (feet), the hips' height puts this rig's lowest foot as high above the ground as the other's, scaled.
"""
import json, sys

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

args = sys.argv[sys.argv.index("--") + 1:]
map_path, src_info_path, src_motion_path, info_path, out_path = args[:5]
tmpl = json.load(open(map_path))
src_info, m, info = json.load(open(src_info_path)), json.load(open(src_motion_path)), json.load(open(info_path))
T = len(m["euler"])

# the other rig: rests from its rig_info, posed orientations from its motion (P_b = P_parent A_parent^T A_b L_b)
s_order = m["bones"]
s_bones = {b["name"]: b for b in src_info["bones"]}
s_names = [b["name"] for b in src_info["bones"]]
As = {n: np.array(s_bones[n]["matrix_local"])[:3, :3] for n in s_order}
s_parent = {n: (s_names[s_bones[n]["parent"]] if s_bones[n]["parent"] >= 0 else None) for n in s_order}
Ps = {}
for i, n in enumerate(s_order):  # the motion lists parents before children
    if i == 0:
        L = np.array([Quaternion(q).to_matrix() for q in m["root_quat"]])
    else:
        L = np.array([Euler(e, "XYZ").to_matrix() for e in np.array(m["euler"])[:, i]])
    p = s_parent[n]
    Ps[n] = np.einsum("ab,tbc->tac", As[n], L) if p is None else \
        np.einsum("tab,cb,cd,tde->tae", Ps[p], As[p], As[n], L)
Ds = {n: np.einsum("tab,cb->tac", Ps[n], As[n]) for n in s_order}

# this rig
rig = bpy.data.objects["rig"]
order = [b["name"] for b in info["bones"]]
parent = {b["name"]: (order[b["parent"]] if b["parent"] >= 0 else None) for b in info["bones"]}
A = {n: np.array(rig.data.bones[n].matrix_local.to_3x3()) for n in order}
eye = np.repeat(np.eye(3)[None], T, 0)
Db = {}
for n in order:
    Db[n] = Ds[n] if n in Ds else (Db[parent[n]] if parent[n] else eye)
for n in tmpl.get("aim", {}):
    if n not in Ds or n not in A:
        continue
    P = np.einsum("tab,bc->tac", Db[n], A[n])
    u = Ps[n][:, :, 1]  # where the other rig's bone points
    for t in range(T):
        P[t] = np.array(Vector(P[t][:, 1]).rotation_difference(Vector(u[t])).to_matrix()) @ P[t]
    Db[n] = np.einsum("tab,cb->tac", P, A[n])

# pose basis, as retarget_kimodo.py: L_b = A_b^T D_parent^T D_b A_b
euler = np.zeros((T, len(order), 3))
root_quat = np.zeros((T, 4))
for i, n in enumerate(order):
    Dp = Db[parent[n]] if parent[n] else eye
    L = np.einsum("ba,tcb,tcd,de->tae", A[n], Dp, Db[n], A[n])
    prev = None
    for t in range(T):
        mm = Matrix(L[t].tolist())
        if parent[n] is None:
            root_quat[t] = np.array(mm.to_quaternion())
        else:
            prev = mm.to_euler("XYZ", prev) if prev is not None else mm.to_euler("XYZ")
            euler[t, i] = np.array(prev)

s = rig.data.bones[order[0]].head_local.z / s_bones[s_order[0]]["matrix_local"][2][3]
trans = np.array(m["trans"], float) * s

pr = rig.pose.bones[order[0]]
for pb in rig.pose.bones:
    pb.rotation_mode = "XYZ"
pr.rotation_mode = "QUATERNION"
B = rig.data.bones[order[0]].matrix_local.to_3x3()
joints = np.zeros((T, len(info["keypoints"]), 3))


def pose(t):
    pr.rotation_quaternion = root_quat[t].tolist()
    pr.location = B.transposed() @ Vector(trans[t])
    for i, n in enumerate(order[1:], 1):
        rig.pose.bones[n].rotation_euler = euler[t, i].tolist()
    bpy.context.view_layer.update()
    for k, kp in enumerate(info["keypoints"]):
        pb = rig.pose.bones[order[kp["bone"]]]
        joints[t, k] = np.array(rig.matrix_world @ (pb.head if kp["end"] == "head" else pb.tail))


for t in range(T):
    pose(t)
is_foot = lambda name: name.split("_")[0] in tmpl.get("feet", [])
feet = [k for k, kp in enumerate(info["keypoints"]) if is_foot(kp["name"])]
s_feet = [k for k, name in enumerate(m.get("joint_names", [])) if is_foot(name)]
if feet and s_feet and "joints" in m:
    s_low = np.array(m["joints"], float)[:, s_feet, 2].min(1) * s
    trans[:, 2] += s_low - joints[:, feet, 2].min(1)
    for t in range(T):
        pose(t)

motion = {**{k: v for k, v in m.items() if k not in ("joints", "joint_names")},
          "bones": order, "trans": trans.round(5).tolist(), "root_quat": root_quat.round(6).tolist(),
          "euler": euler.round(5).tolist(), "joints": joints.round(5).tolist(),
          "joint_names": [k["name"] for k in info["keypoints"]], "retargeted_from": m.get("source")}
json.dump(motion, open(out_path, "w"))
print(f"retargeted {T} frames from {len(s_order)} bones onto {len(order)} (hips scaled {s:.3f}) -> {out_path}")
