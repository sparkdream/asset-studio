"""Drive a rig from a Kimodo motion (rigging.kimodo_gen): each bone turns in the world as its SOMA joint does.

blender -b rigged.blend -P rigging/blender/retarget_kimodo.py -- MAP.json KIMODO.npz rig_info.json OUT_motion.json
    [--speed 1.0] [--smooth 0]

MAP.json: rig bone -> SOMA joint (rigging/templates/humanoid_kimodo.json).

Kimodo's rotations are already world turns since SOMA's rest pose, a T-pose. The rig's rest pose is the character's
own, so each limb's rest is taken as SOMA's rest limb turned the least onto the rig's bone (G_rest = S), and each
frame's turn is D = G_frame S^T, applied to the bone's rest orientation. Limbs are then aimed along SOMA's joint
positions, as retarget.py does with SAM's. Kimodo knows nothing of held props: they turn with the hand.

The hips move as SOMA's do across the ground, scaled by the rig's height; their height comes from the feet, as in
retarget.py. motion.json has the layout apply_motion.py reads.
"""
import json, math, sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
map_path, clip_path, info_path, out_path = args[:4]
opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d
speed, smooth = opt("--speed", 1.0), opt("--smooth", 0.0)

tmpl = json.load(open(map_path))
info = json.load(open(info_path))
clip = np.load(clip_path)
names = [str(n) for n in clip["names"]]
J = {n: i for i, n in enumerate(names)}

# Kimodo's axes (y up, facing +z, x to the character's left) -> Blender's (z up, facing -y, x to its left)
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
G = np.einsum("ab,tjbc,dc->tjad", M, clip["rots"], M)
P3 = np.einsum("ab,tjb->tja", M, clip["joints"])
rest = clip["rest"] @ M.T
T = len(G)
print("Kimodo:", str(clip["text"])[:80], f"| {T} frames at {float(clip['fps'])} fps",
      "| left hand at x =", round(float(rest[J["LeftHand"], 0]), 2))


def smooth_rots(R, sigma):
    """Gaussian smoothing over time of one joint's rotations, as sign-aligned quaternions."""
    if sigma <= 0:
        return R
    q = np.array([np.array(Matrix(r.tolist()).to_quaternion()) for r in R])
    for t in range(1, len(q)):
        if q[t] @ q[t - 1] < 0:
            q[t] = -q[t]
    k = np.exp(-0.5 * (np.arange(-3 * int(math.ceil(sigma)), 3 * int(math.ceil(sigma)) + 1) / sigma) ** 2)
    pad = len(k) // 2
    qp = np.concatenate([q[:1].repeat(pad, 0), q, q[-1:].repeat(pad, 0)])
    qs = np.array([np.convolve(qp[:, i], k / k.sum(), "valid") for i in range(4)]).T
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    from mathutils import Quaternion
    return np.array([np.array(Quaternion(x).to_matrix()) for x in qs])


# the rig: each bone's rest orientation in world axes
rig = bpy.data.objects["rig"]
order = [b["name"] for b in info["bones"]]
parent = {b["name"]: (order[b["parent"]] if b["parent"] >= 0 else None) for b in info["bones"]}
A = {n: np.array(rig.data.bones[n].matrix_local.to_3x3()) for n in order}

# each joint's rest in the rig's pose: SOMA's rest limb turned the least onto the rig's bone (the first bone aimed
# along that joint's limb); the torso and head stand upright in both, so theirs is SOMA's own
S = {}
for n, (ja, jb) in tmpl.get("aim", {}).items():
    j = tmpl["map"][n]
    if j not in S:
        u = rest[J[jb]] - rest[J[ja]]
        S[j] = np.array(Vector(u).rotation_difference(Vector(A[n][:, 1])).to_matrix())
D = {}
for j in sorted(set(tmpl["map"].values())):
    D[j] = np.einsum("tab,cb->tac", smooth_rots(G[:, J[j]], smooth), S.get(j, np.eye(3)))  # G_t S^T

Db = {}
for n in order:  # rig_info lists parents before children
    Db[n] = D[tmpl["map"][n]] if n in tmpl["map"] else (Db[parent[n]] if parent[n] else np.repeat(np.eye(3)[None], T, 0))

# limbs aimed along SOMA's joint positions each frame (minimal turn, so the roll is kept)
for n, (ja, jb) in tmpl.get("aim", {}).items():
    u = P3[:, J[jb]] - P3[:, J[ja]]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = np.einsum("tab,bc->tac", Db[n], A[n])
    for t in range(T):
        P[t] = np.array(Vector(P[t][:, 1]).rotation_difference(Vector(u[t])).to_matrix()) @ P[t]
    Db[n] = np.einsum("tab,cb->tac", P, A[n])
    for c in order:  # bones with no joint of their own (a held blade) follow it
        if c not in tmpl["map"] and parent[c] == n:
            Db[c] = Db[n]

# pose basis: P_b = P_parent A_parent^T A_b L_b and P_b = D_b A_b, so L_b = A_b^T D_parent^T D_b A_b
euler = np.zeros((T, len(order), 3))
root_quat = np.zeros((T, 4))
for i, n in enumerate(order):
    Dp = Db[parent[n]] if parent[n] else np.repeat(np.eye(3)[None], T, 0)
    L = np.einsum("ba,tcb,tcd,de->tae", A[n], Dp, Db[n], A[n])
    prev = None
    for t in range(T):
        m = Matrix(L[t].tolist())
        if parent[n] is None:
            root_quat[t] = np.array(m.to_quaternion())
        else:  # each frame's Euler kept nearest the last, so keyframes in between don't spin the long way round
            prev = m.to_euler("XYZ", prev) if prev is not None else m.to_euler("XYZ")
            euler[t, i] = np.array(prev)

# the hips across the ground as SOMA's, from where they start, scaled to the rig
h_a, h_b = (J[n] for n in tmpl["height_joints"])
h_rig = (rig.data.bones[tmpl["height_bones"][1]].head_local - rig.data.bones[tmpl["height_bones"][0]].head_local).length
s = h_rig / np.linalg.norm(rest[h_b] - rest[h_a])
hips = P3[:, J[tmpl["map"][order[0]]]]
trans = np.zeros((T, 3))
trans[:, :2] = (hips[:, :2] - hips[0, :2]) * s

# joints for the studio's checks: the rig posed, each keypoint read off its bone
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


# the hips' height: the rig's lowest foot joint as high above the ground as SOMA's lowest foot is, scaled to the rig.
# On the ground that puts the rig's feet where they stand at rest (the rig's legs aren't SOMA's length, so its hips
# can't simply copy SOMA's); in the air it keeps the jump. Only the ground-contact part is smoothed.
feet = [k for k, kp in enumerate(info["keypoints"]) if kp["name"].split("_")[0] in tmpl.get("feet", [])]
rest_low = min(kp["rest"][2] for kp in info["keypoints"] if kp["name"].split("_")[0] in tmpl.get("feet", []))
for t in range(T):
    pose(t)
if feet:
    soma_low = P3[:, [J[n] for n in ("LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase") if n in J]].min(1)[:, 2]
    clear = np.clip(soma_low - np.percentile(soma_low, 5), 0, None) * s
    clear[clear < 0.02] = 0  # contact jitter stays on the ground
    lift = rest_low - joints[:, feet, 2].min(1)
    trans[:, 2] = np.convolve(np.pad(lift, 2, mode="edge"), np.ones(5) / 5, "valid") + clear
    if clear.max() > 0.05:
        print(f"airborne: up to {clear.max():.2f} m for {int((clear > 0.05).sum())} frames")
    for t in range(T):
        pose(t)
    # planted feet stay put: a foot on the ground (its lowest point within 2.5 cm of where the feet stand at rest) that
    # the motion moves across the ground drags the whole body back by that drift. The hips cross the ground as SOMA's
    # do, scaled, but the rig's legs aren't SOMA's, so planted feet slid (14-21 cm in the rogue's leap, crouching and
    # landing). Measured on the contact point (a foot's lowest keypoint the frame before: a heel lifts, the ball stays).
    sides = {}
    for k in feet:
        sides.setdefault(info["keypoints"][k]["name"].rsplit("_", 1)[-1], []).append(k)
    corr = np.zeros((T, 3))
    for t in range(1, T):
        drift = []
        for ks in sides.values():
            prev = ks[int(np.argmin(joints[t - 1, ks, 2]))]
            if joints[t - 1, prev, 2] < rest_low + 0.025 and joints[t, ks, 2].min() < rest_low + 0.025:
                drift.append(joints[t, prev, :2] - joints[t - 1, prev, :2])
        corr[t, :2] = corr[t - 1, :2] - (np.mean(drift, 0) if drift else 0)
    if np.abs(corr).max() > 0.005:
        print(f"planted feet: the body held back up to {np.linalg.norm(corr[:, :2], axis=1).max() * 100:.0f} cm")
        trans += corr
        for t in range(T):
            pose(t)

fps = float(clip["fps"])
motion = {"fps": fps * speed, "speed": speed, "bones": order, "trans": trans.round(5).tolist(),
          "root_quat": root_quat.round(6).tolist(), "euler": euler.round(5).tolist(), "joints": joints.round(5).tolist(),
          "joint_names": [k["name"] for k in info["keypoints"]], "source": "kimodo_retarget",
          "text": str(clip["text"])}
json.dump(motion, open(out_path, "w"))
print(f"retargeted {T} frames onto {len(order)} bones -> {out_path}")
