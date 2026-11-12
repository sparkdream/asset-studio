"""Drive a rig from SAM 3D Body's reading of a clip: each bone turns in the world as its MHR joint does.

blender -b rigged.blend -P rigging/blender/retarget.py -- MAP.json REST_DIR SAM.npz rig_info.json OUT_motion.json
    [--speed 1.5] [--smooth 1.0]

MAP.json: rig bone -> MHR joint (rigging/templates/*.json). REST_DIR: render_rest.py's renders and rest_cams.json, with
SAM's reading of them in rest_sam.npz. SAM.npz: rigging.sam_body on the clip.

The rig's rest pose needn't be MHR's: SAM reads the character's own rest pose from the renders, and each frame's
joint rotation is taken relative to that reading (D = G_frame G_rest^T, in world axes), then applied to the bone's rest
orientation. So the rig's bone axes, rolls and proportions never matter, and errors SAM makes on this character's
build mostly cancel. The joint rotations come from MHR's anatomical model: a hinge elbow and knee, the forearm's own
twist, joint limits learned from real people.

The hips move where SAM's hip keypoints are in the front view (orthographic, so pixels are metres) and as deep as
SAM puts them. motion.json has the layout apply_motion.py reads.
"""
import json, math, sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:]
map_path, rest_dir, sam_path, info_path, out_path = args[:5]
opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d
speed, smooth = opt("--speed", 1.0), opt("--smooth", 1.0)

tmpl = json.load(open(map_path))
info = json.load(open(info_path))
lay = info["layout"]
cams = json.load(open(f"{rest_dir}/rest_cams.json"))["cams"]
rest = np.load(f"{rest_dir}/rest_sam.npz")
clip = np.load(sam_path)
names = [str(n) for n in clip["names"]]
kpn = [str(n) for n in clip["kp_names"]] if "kp_names" in clip else None
J = {n: i for i, n in enumerate(names)}


def mean_rot(Rs, w):
    """The rotation closest to the weighted average of several (chordal mean)."""
    u, _, vt = np.linalg.svd(np.einsum("v,vab->ab", w, Rs))
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1, 1, d]) @ vt


# SAM's reading of the rest pose, in world axes, averaged over the views
Cs = np.array([c["rot"] for c in cams])
G_views = np.einsum("vab,vjbc->vjac", Cs, rest["rots"])
# the views turned furthest are where SAM errs most, and the clips are filmed from the front: cos^4 of the turn
wv = np.array([math.cos(math.radians(c["yaw"])) ** 4 for c in cams])
G_rest = np.array([mean_rot(G_views[:, j], wv) for j in range(len(names))])
spread = {n: max(math.degrees(math.acos(np.clip((np.trace(G_views[v, J[n]].T @ G_rest[J[n]]) - 1) / 2, -1, 1)))
                 for v in range(len(cams)) if abs(cams[v]["yaw"]) <= 30) for n in set(tmpl["map"].values())}
print("rest reading, worst view within 30 deg of the front off the mean (deg):", {k: round(v) for k, v in sorted(spread.items(), key=lambda x: -x[1])[:6]})

# each frame in world axes: the clips' front camera looks along +Y, like render_rest's yaw 0
C = np.array(next(c for c in cams if c["yaw"] == 0)["rot"])
G = np.einsum("ab,tjbc->tjac", C, clip["rots"])
T = len(G)


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


used = sorted(set(tmpl["map"].values()))
D = {}
for n in used:
    D[n] = np.einsum("tab,cb->tac", smooth_rots(G[:, J[n]], smooth), G_rest[J[n]])  # G_t G_rest^T

# the rig: each bone's rest orientation in world axes, and the turn D it takes (its own joint's, else its parent's)
rig = bpy.data.objects["rig"]
order = [b["name"] for b in info["bones"]]
parent = {b["name"]: (order[b["parent"]] if b["parent"] >= 0 else None) for b in info["bones"]}
A = {n: np.array(rig.data.bones[n].matrix_local.to_3x3()) for n in order}
Db = {}
for n in order:  # rig_info lists parents before children
    Db[n] = D[tmpl["map"][n]] if n in tmpl["map"] else (Db[parent[n]] if parent[n] else np.repeat(np.eye(3)[None], T, 0))

# Limbs are aimed along SAM's own joint positions each frame (minimal turn, so the roll is kept): a turn taken
# against the rest reading carries that reading's errors into every frame (an elbow SAM sees 30 degrees straighter
# at rest than the rig is would stay 30 degrees bent), and where a limb points is what SAM reads best.
P3 = np.einsum("ab,tjb->tja", C, clip["joints"])
for n, (ja, jb) in tmpl.get("aim", {}).items():
    u = P3[:, J[jb]] - P3[:, J[ja]]
    if smooth > 0:
        k = np.exp(-0.5 * (np.arange(-3 * int(math.ceil(smooth)), 3 * int(math.ceil(smooth)) + 1) / smooth) ** 2)
        k /= k.sum()
        u = np.stack([np.convolve(np.pad(u[:, i], len(k) // 2, mode="edge"), k, "valid") for i in range(3)], 1)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    P = np.einsum("tab,bc->tac", Db[n], A[n])
    for t in range(T):
        S = np.array(Vector(P[t][:, 1]).rotation_difference(Vector(u[t])).to_matrix())
        P[t] = S @ P[t]
    Db[n] = np.einsum("tab,cb->tac", P, A[n])
    for c in order:  # bones with no joint of their own (a held blade) follow it
        if c not in tmpl["map"] and parent[c] == n:
            Db[c] = Db[n]

# Held props: SAM never sees them, so where one points comes from the video. Each frame the bone holding it (the hand)
# turns as little as it must to lay the prop along its 2D direction in the front view, read by a keypoint reader
# (--props DET.npz with base and tip points, e.g. handtip_L and bladetip_L). Direction only: the video model draws
# the character at its own size, so absolute pixels don't match the rig. One view can't tell a prop along the arm from
# one pointing toward the camera; "along" (in the map) leans toward the first, for a blade held forward.
heads = {n: np.array(rig.data.bones[n].head_local) for n in order}
tails = {n: np.array(rig.data.bones[n].tail_local) for n in order}


def fk_point(t, n, p_rest):
    """Where a rest-pose point riding on bone n is in frame t, with the hips at the origin."""
    chain = [n]
    while parent[chain[-1]]:
        chain.append(parent[chain[-1]])
    x = heads[chain[-1]].copy()
    for a, b in zip(chain[::-1], chain[::-1][1:]):
        x = x + Db[a][t] @ (heads[b] - heads[a])
    return x + Db[n][t] @ (p_rest - heads[n])


def rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


if "--props" in args:
    det = np.load(args[args.index("--props") + 1])
    dn = [str(x) for x in det["names"]]
    for prop, spec in tmpl.get("props", {}).items():
        hand = parent[prop]
        bi, ti = dn.index(spec["base"]), dn.index(spec["tip"])
        kb = next(k for k in info["keypoints"] if k["name"] == spec["base"])
        kt = next(k for k in info["keypoints"] if k["name"] == spec["tip"])
        conf = np.minimum(det["front_conf"][:, bi], det["front_conf"][:, ti])
        d2 = det["front_uv"][:, ti] - det["front_uv"][:, bi]
        W = np.zeros((T, 3))
        for t in range(T):
            if conf[t] < 0.3 or np.linalg.norm(d2[t]) < 8:  # unread, or pointing at the camera
                continue
            want = np.array([d2[t, 0], -d2[t, 1]]) / np.linalg.norm(d2[t])  # image (u, v) -> world (x, z)
            pivot = fk_point(t, hand, heads[hand])
            base = fk_point(t, order[kb["bone"]], np.array(kb["rest"])) - pivot
            tip = fk_point(t, order[kt["bone"]], np.array(kt["rest"])) - pivot

            arm = pivot - fk_point(t, parent[hand], heads[parent[hand]])
            arm /= np.linalg.norm(arm)

            def cost(w):
                R = rotvec(w)
                b3 = R @ tip - R @ base
                v = b3[[0, 2]]
                along = 1 - b3 @ arm / max(np.linalg.norm(b3), 1e-9)  # a blade held forward lies along the forearm
                return (1 - v @ want / max(np.linalg.norm(v), 1e-9)) + 0.1 * (w @ w) + spec.get("along", 0) * along

            w = np.zeros(3)
            for _ in range(150):
                g = np.array([(cost(w + e) - cost(w - e)) / 2e-4 for e in np.eye(3) * 1e-4])
                w -= 0.3 * g
                n_ = np.linalg.norm(w)
                if n_ > math.radians(70):
                    w *= math.radians(70) / n_
            W[t] = w * min(1.0, conf[t] / 0.6)
        if smooth > 0:
            W = np.stack([np.convolve(np.pad(W[:, i], 3, mode="edge"), np.ones(7) / 7, "valid") for i in range(3)], 1)
        Db[hand] = np.einsum("tab,tbc->tac", np.array([rotvec(w) for w in W]), Db[hand])
        for c in order:  # what it holds turns with it
            if parent[c] == hand and c not in tmpl["map"]:
                Db[c] = Db[hand]
        print(f"{prop}: the hand turned to follow it, mean {np.degrees(np.linalg.norm(W, axis=1)).mean():.0f} deg, "
              f"most {np.degrees(np.linalg.norm(W, axis=1)).max():.0f}")

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

# the hips: across and up from the front view's pixels (orthographic), depth from SAM, all against the rest reading
hk = [kpn.index(k) for k in tmpl["hips_keypoints"]] if kpn else [9, 10]
hp = clip["kp2d"][:, hk].mean(1)
m_px = lay["M_PER_PX"]
x = (hp[:, 0] - lay["W_FRONT"] / 2) * m_px
z = lay["Z_CENTER"] + (lay["H"] / 2 - hp[:, 1]) * m_px
c0 = next(c for c in cams if c["yaw"] == 0)
r0 = rest["kp2d"][cams.index(c0), hk].mean(0)
px = c0["ortho_scale"] / c0["res"][1]
x0 = c0["centre"][0] + (r0[0] - c0["res"][0] / 2) * px
z0 = c0["centre"][2] + (c0["res"][1] / 2 - r0[1]) * px
h_a, h_b = (J[n] for n in tmpl["height_joints"])
h_sam = np.linalg.norm(rest["joints"][:, h_b] - rest["joints"][:, h_a], axis=1).mean()
h_rig = (rig.data.bones[tmpl["height_bones"][1]].head_local - rig.data.bones[tmpl["height_bones"][0]].head_local).length
s = h_rig / h_sam
depth = -(clip["joints"][:, J["root"], 2] - np.median(clip["joints"][:, J["root"], 2])) * s  # camera z is world -y
depth = np.convolve(np.pad(depth, 3, mode="edge"), np.ones(7) / 7, "valid")
trans = np.stack([x - x0, depth, np.zeros(T)], 1)  # height: set below, from the feet

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


# The video model draws the character at its own size, so pixel heights don't say how high the hips are. Each frame
# the hips sit where the lowest foot joint is as high as it is at rest (standing moves: a jump would need the pixels).
feet = [k for k, kp in enumerate(info["keypoints"]) if kp["name"].split("_")[0] in tmpl.get("feet", [])]
rest_low = min(kp["rest"][2] for kp in info["keypoints"] if kp["name"].split("_")[0] in tmpl.get("feet", []))
for t in range(T):
    pose(t)
if feet:
    lift = rest_low - joints[:, feet, 2].min(1)
    trans[:, 2] = np.convolve(np.pad(lift, 2, mode="edge"), np.ones(5) / 5, "valid")
    for t in range(T):
        pose(t)

fps = float(clip["fps"]) or 24.0
motion = {"fps": fps * speed, "speed": speed, "bones": order, "trans": trans.round(5).tolist(),
          "root_quat": root_quat.round(6).tolist(), "euler": euler.round(5).tolist(), "joints": joints.round(5).tolist(),
          "joint_names": [k["name"] for k in info["keypoints"]], "source": "sam3d_retarget",
          "rest_spread_deg": {k: round(v, 1) for k, v in spread.items()}}
json.dump(motion, open(out_path, "w"))
print(f"retargeted {T} frames onto {len(order)} bones -> {out_path}")
