"""A Kimodo capture with its posture changed, as a new capture: the upper body tilted forward or back at the waist,
and/or both arms swept straight back (a ninja's sprint). For what words can't get out of Kimodo: its standing poses
lean back a few degrees (the rogue's idle stood arched), and no prompt made it run with the arms trailing.

    python -m rigging.kimodo_posture PROJECT_DIR CAPTURE --name NAME [--lean DEG] [--tuck DEG] [--arms-back DEG]
        [--pump RATIO [--elbow DEG] [--elbow-out DEG] [--pump-forward DEG]] [--feet-forward] [--head SHARE] [--move MOVE]
        [--blade-up SIDE:FROM-TO[:BACK_DEG[:OUT_DEG]]] [--blade-slice SIDE:FROM-TO] [--strike-path SIDE:FROM-TO[:ARC_CM]]

CAPTURE is a capture folder holding kimodo.npz (in PROJECT_DIR's anymove/runs/clips, or a path to one anywhere, e.g.
another project's). Changed on Kimodo's motion itself (every joint's world rotation and position, y up), before the
retarget, so the rig follows as it follows any Kimodo motion and its feet are grounded as usual:
- --lean: the upper body turned forward (+) or back (-) about the character's own left-right axis, half at Spine1,
  0.3 at Spine2 and 0.2 at the Chest: most of it low, so a sway-back straightens and a runner's whole torso tips
  (spread evenly, the rogue's sprint bent at the neck instead); the legs and hips stay.
- --head: the neck turned back by this share of the lean (default 0.5), so a leaning character still looks ahead.
- --tuck: each upper arm turned in toward the body by DEG at the shoulder (about the character's forward axis), so
  flared elbows come in (a run whose elbows stuck out 2.3 times as far as the shoulders read as silly). On a swinging
  arm this traces an arc (the rogue's right arm went round in circles): for a run use --pump instead.
- --feet-forward: knees over the feet and feet straight ahead. Each leg turned at the hip so its heel swings no
  further out than the hip (a foot kicked up behind flared out past it), then about its hip-to-ankle line until the
  knee points the way the character faces (measured while it bends), then each foot turned about its shin until it
  sits in line with the knee as at rest; smoothed over time. Planted feet land inside the hip's line, so where they
  stand is unchanged (and with it the grounding). The rogue's run turned her left foot 59 deg out in every swing;
  aiming the toes themselves ahead buckled her knees in instead.
- --pump RATIO: both arms rebuilt as a runner's, from the legs, the shoulders level and alike (each collarbone as at
  rest on the chest, not wherever the motion held it): each upper arm swings fore and aft in the torso's own
  plane, RATIO times as far as the opposite thigh (arm and opposite leg move together), --pump-forward DEG ahead of
  hanging on average; the elbow bent to --elbow DEG (inner angle; the forearm hinges about the elbow's left-right
  axis, so nothing twists) and held --elbow-out DEG out from the body; wrists straight, fingers keep their grip.
- --arms-back: each arm held straight and pointing back, DEG below horizontal, a little out from the body, wherever
  the torso goes (the character's heading, not its lean); fingers keep their grip.
- --blade-up SIDE:FROM-TO[:BACK_DEG[:OUT_DEG]] (SIDE R or L, seconds; repeatable): from FROM to TO the item in that
  hand points up, tipped BACK_DEG back from vertical (default 45: a chop's wind-up) and OUT_DEG out to that side
  (default 0), easing in over the window's first 0.12 s and back to the motion's
  own over its last (so a strike's snap is Kimodo's). The wrist turns the least, bending no more than 50 deg; the
  forearm tilts at the elbow for the rest (overhead, a fist-held blade can't point up from the wrist alone). Where the item
  points in the hand comes from the rig's hand socket (rigging/blender/socket_axes.py) through the retarget's rest
  match. The rogue's chop wound up with the dagger lying flat, pointing in at her head.
- --blade-slice SIDE:FROM-TO (repeatable, applied after --blade-up): from FROM to TO a held blade cuts edge first: its
  flat turned square to the swing (the plane of the arm and the hand's path) and its tip out along the arm, eased in
  and out like --blade-up and only while the hand moves (below 0.8 m/s the motion's own grip shows through). The
  same wrist/elbow split. The rogue's chops hit with the dagger's flat, as if batting.
- --strike-path SIDE:FROM-TO[:ARC_CM] (applied last): from FROM to TO the hand travels the straight line between where
  it is at those times, bowed ARC_CM (default 4) out to that side, keeping its own pace along it (eased in and out over
  0.08 s); the hand keeps its turn (so the blade its angle) and the elbow swings round the shoulder-hand line to
  wherever the wrist bends least, near its own place. The rogue's chop swung round in a 27 cm arc to the outside.
Written to PROJECT_DIR/anymove/runs/clips/NAME/ and exported on the project's rig, like any capture.
"""
import argparse, json, subprocess, time
from pathlib import Path

import numpy as np

from rigging.kimodo_capture import BLENDER, ROOT, run


def axis_angle(axis, angle):
    """Rotation matrices (T,3,3) about unit axes (T,3) by angles (T,) or a scalar (radians)."""
    angle = np.broadcast_to(angle, axis.shape[:1])
    K = np.zeros(axis.shape[:1] + (3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = axis[:, 2], -axis[:, 1], axis[:, 0]
    return np.eye(3) + np.sin(angle)[:, None, None] * K + (1 - np.cos(angle))[:, None, None] * K @ K


def turn_onto(u, v):
    """The least rotation (T,3,3) taking unit vectors u onto v."""
    axis = np.cross(u, v)
    s = np.linalg.norm(axis, axis=1)
    c = np.einsum("ti,ti->t", u, v)
    axis = np.where(s[:, None] > 1e-9, axis / np.maximum(s, 1e-9)[:, None], np.array([1.0, 0, 0]))
    return axis_angle(axis, np.arctan2(s, c))


def subtree(parents, root):
    out, kids = [root], {}
    for j, p in enumerate(parents):
        kids.setdefault(int(p), []).append(j)
    i = 0
    while i < len(out):
        out += kids.get(out[i], [])
        i += 1
    return out


LEAN_SHARES = {"Spine1": 0.5, "Spine2": 0.3, "Chest": 0.2}


def posture(z, lean_deg=0.0, arms_back_deg=None, head=0.5, tuck_deg=0.0, pump=None, elbow_deg=90.0, elbow_out_deg=8.0,
            pump_forward_deg=10.0, feet_forward=False, blades=(), wrist_max=50.0, paths=()):
    names = [str(n) for n in z["names"]]
    J = {n: i for i, n in enumerate(names)}
    parents = z["parents"]
    R, P, rest = z["rots"].copy(), z["joints"].copy(), z["rest"]
    f = R[:, J["Hips"], :, 2]  # the hips' forward (+z at rest), flattened to a heading
    yaw = np.arctan2(f[:, 0], f[:, 2])
    left = np.stack([np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)], 1)  # the character's left (+x at rest)
    fwd = np.stack([np.sin(yaw), np.zeros_like(yaw), np.cos(yaw)], 1)
    def tilt(k, deg):  # +deg about +left takes up (+y) toward forward
        Q = axis_angle(left, np.radians(deg))
        pivot = P[:, J[k]].copy()
        for j in subtree(parents, J[k]):
            R[:, j] = Q @ R[:, j]
            P[:, j] = pivot + np.einsum("tab,tb->ta", Q, P[:, j] - pivot)

    # the way she travels: the hips' path (smoothed over ~a stride), or where the hips face when barely moving.
    # Not the hips' own heading: they twist each step (and averaged still faced a little off the path)
    kh = np.exp(-0.5 * (np.arange(-30, 31) / 10.0) ** 2)
    sm = lambda x: np.convolve(np.pad(x, 30, mode="edge"), kh / kh.sum(), "valid")
    vel = np.gradient(P[:, J["Hips"]], axis=0) * float(z["fps"])
    vx, vz = sm(vel[:, 0]), sm(vel[:, 2])
    w = np.clip(np.hypot(vx, vz) / 0.5, 0, 1)
    hc, hs = sm(np.cos(yaw)), sm(np.sin(yaw))
    hy = np.arctan2(w * vx / np.maximum(np.hypot(vx, vz), 1e-6) + (1 - w) * hs,
                    w * vz / np.maximum(np.hypot(vx, vz), 1e-6) + (1 - w) * hc)
    travel_left = np.stack([np.cos(hy), np.zeros_like(hy), -np.sin(hy)], 1)
    travel_fwd = np.stack([np.sin(hy), np.zeros_like(hy), np.cos(hy)], 1)

    if lean_deg:
        for k, share in LEAN_SHARES.items():
            tilt(k, lean_deg * share)
        if head:
            tilt("Neck1", -lean_deg * head)
    if tuck_deg:  # about the forward axis at each shoulder: the left arm (+left side) by -deg, the right by +deg
        for side, sign in (("Left", -1.0), ("Right", 1.0)):
            Q = axis_angle(fwd, sign * np.radians(tuck_deg))
            pivot = P[:, J[f"{side}Arm"]].copy()
            for j in subtree(parents, J[f"{side}Arm"]):
                R[:, j] = Q @ R[:, j]
                P[:, j] = pivot + np.einsum("tab,tb->ta", Q, P[:, j] - pivot)
    if feet_forward:
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
        k /= k.sum()

        def about(ax, v, w):  # the angle about unit axes ax turning v onto w (both flattened square to ax), and |v|
            v = v - np.einsum("ti,ti->t", v, ax)[:, None] * ax
            w = w - np.einsum("ti,ti->t", w, ax)[:, None] * ax
            ang = np.arctan2(np.einsum("ti,ti->t", np.cross(v, w), ax), np.einsum("ti,ti->t", v, w))
            return ang, np.linalg.norm(v, axis=1) * (np.linalg.norm(w, axis=1) > 1e-3)

        def smooth(ang, wgt):  # weighted over time: frames where the angle is ill-defined borrow their neighbours'
            c, s = np.cos(ang) * wgt, np.sin(ang) * wgt
            c, s = (np.convolve(np.pad(x, 6, mode="edge"), k, "valid") for x in (c, s))
            return np.where(np.hypot(c, s) > 1e-6, np.arctan2(s, c), 0.0)

        def turn(j0, pivot, Q):
            pivot = pivot.copy()
            for j in subtree(parents, j0):
                R[:, j] = Q @ R[:, j]
                P[:, j] = pivot + np.einsum("tab,tb->ta", Q, P[:, j] - pivot)

        for side, out_sign in (("Left", 1.0), ("Right", -1.0)):
            hip, knee, ankle, toe = (J[f"{side}{n}"] for n in ("Leg", "Shin", "Foot", "ToeBase"))
            # the heel no further out than the hip: the leg turned at the hip (the least) until the ankle is back on
            # the hip's line. A foot kicked up behind flared out past it (the rogue's heels 8 cm out at every back
            # kick); planted feet land under the body, inside that line, so they are left alone
            # "out" across the way she travels (the hips' heading averaged over ~a stride: the hips twist each step,
            # and with the foot 40 cm behind a 10 deg twist moves it 7 cm sideways)
            out = out_sign * travel_left
            v = P[:, ankle] - P[:, hip]
            excess = np.maximum(np.einsum("ti,ti->t", v, out), 0.0)
            excess = np.max([np.roll(np.pad(excess, 3, mode="edge"), s)[3:-3] for s in range(-3, 4)], 0)  # peaks kept
            excess = np.convolve(np.pad(excess, 6, mode="edge"), k, "valid")
            target = v - excess[:, None] * out
            turn(hip, P[:, hip], turn_onto(v / np.linalg.norm(v, axis=1, keepdims=True),
                                           target / np.linalg.norm(target, axis=1, keepdims=True)))
            # the knee aimed ahead: the leg turned about its hip-ankle line (the ankle stays). Only a bent knee says
            # where it points, so straight frames take their neighbours' turn
            ax = P[:, ankle] - P[:, hip]
            ax /= np.linalg.norm(ax, axis=1, keepdims=True)
            ang, bend = about(ax, P[:, knee] - P[:, hip], travel_fwd)
            turn(hip, P[:, hip], axis_angle(ax, smooth(ang, np.clip(bend / 0.08, 0, 1))))
            # the foot in line with the knee, as at rest: turned about the shin (the ankle stays) until the toes point
            # where the shin's rest foot would. Not toward the heading: a foot swung up behind points down, and
            # aiming it ahead twisted the whole leg, knee in (the rogue's run buckled 25-30 deg at every swing)
            sh = P[:, ankle] - P[:, knee]
            sh /= np.linalg.norm(sh, axis=1, keepdims=True)
            toe_rest = np.einsum("tab,b->ta", R[:, knee], rest[toe] - rest[ankle])
            ang, size = about(sh, P[:, toe] - P[:, ankle], toe_rest)
            turn(ankle, P[:, ankle], axis_angle(sh, smooth(ang, size)))
    if pump is not None:
        # the upper body squared to the path: its twist each step kept, its average turn off the path taken out at
        # the waist (the rogue's chest swung 22 deg one way and 12 the other, so one arm crossed the body more).
        # Averaged with sigma 0.35 s (half a stride), which keeps the twist itself but not the run speeding up or slowing
        fc = R[:, J["Chest"], :, 2]
        ks = np.exp(-0.5 * (np.arange(-45, 46) / (0.35 * float(z["fps"]))) ** 2)
        sm2 = lambda x: np.convolve(np.pad(x, 45, mode="edge"), ks / ks.sum(), "valid")
        cy = np.arctan2(sm2(fc[:, 0]), sm2(fc[:, 2]))
        off = np.angle(np.exp(1j * (cy - hy)))
        Q = axis_angle(np.repeat(np.array([[0.0, 1.0, 0.0]]), len(R), 0), -off)
        pivot = P[:, J["Spine1"]].copy()
        for j in subtree(parents, J["Spine1"]):
            R[:, j] = Q @ R[:, j]
            P[:, j] = pivot + np.einsum("tab,tb->ta", Q, P[:, j] - pivot)
        # the torso's own axes, from the Chest's world rotation (identity at rest: x left, y up, z forward)
        Rc = R[:, J["Chest"]]
        tl, tu, tf = Rc[:, :, 0], Rc[:, :, 1], Rc[:, :, 2]
        # the shoulders level and alike: each collarbone as at rest on the chest (the rogue's run held the left
        # shoulder 2.7 cm lower and 2.3 cm further back than the right, and the arm hung from it rolled in)
        for side in ("Left", "Right"):
            sh = J[f"{side}Shoulder"]
            if parents[sh] == J["Chest"]:
                D = Rc @ np.transpose(R[:, sh], (0, 2, 1))
                pivot = P[:, sh].copy()
                for j in subtree(parents, sh):
                    R[:, j] = D @ R[:, j]
                    P[:, j] = pivot + np.einsum("tab,tb->ta", D, P[:, j] - pivot)
        for side, other, out_sign in (("Left", "Right", 1.0), ("Right", "Left", -1.0)):
            thigh = P[:, J[f"{other}Shin"]] - P[:, J[f"{other}Leg"]]  # the opposite thigh, hip to knee
            swing = np.arctan2(np.einsum("ti,ti->t", thigh, fwd), -thigh[:, 1])  # + forward of hanging
            phi = pump * swing + np.radians(pump_forward_deg)
            out = np.radians(elbow_out_deg)
            d_ua = (np.cos(phi)[:, None] * -tu + np.sin(phi)[:, None] * tf) * np.cos(out) + out_sign * np.sin(out) * tl
            d_ua /= np.linalg.norm(d_ua, axis=1, keepdims=True)
            hinge = np.cross(d_ua, tf + 1e-9)  # the elbow bends in the plane of the arm and forward
            hinge /= np.linalg.norm(hinge, axis=1, keepdims=True)
            bend = np.radians(180.0 - elbow_deg)
            Qb = axis_angle(hinge, -bend)
            d_fa = np.einsum("tab,tb->ta", Qb, d_ua)
            if np.einsum("ti,ti->t", d_fa, tf).mean() < np.einsum("ti,ti->t", d_ua, tf).mean():  # bent the wrong way
                Qb = axis_angle(hinge, bend)
                d_fa = np.einsum("tab,tb->ta", Qb, d_ua)
            arm, fore, hand = J[f"{side}Arm"], J[f"{side}ForeArm"], J[f"{side}Hand"]
            r0 = rest[fore] - rest[arm]
            r0 = np.repeat((r0 / np.linalg.norm(r0))[None], len(R), 0)
            R_ua = turn_onto(r0, d_ua)
            R_fa = turn_onto(d_ua, d_fa) @ R_ua  # the upper arm's frame hinged on: no twist at the elbow
            old_hand_R, old_hand_P = R[:, hand].copy(), P[:, hand].copy()
            R[:, arm], R[:, fore], R[:, hand] = R_ua, R_fa, R_fa
            P[:, fore] = P[:, arm] + np.einsum("tab,b->ta", R_ua, rest[fore] - rest[arm])
            P[:, hand] = P[:, fore] + np.einsum("tab,b->ta", R_fa, rest[hand] - rest[fore])
            D = R_fa @ np.transpose(old_hand_R, (0, 2, 1))
            for j in subtree(parents, hand)[1:]:
                R[:, j] = D @ R[:, j]
                P[:, j] = P[:, hand] + np.einsum("tab,tb->ta", D, P[:, j] - old_hand_P)
    if arms_back_deg is not None:
        drop = np.radians(arms_back_deg)
        for side, out_sign in (("Left", 1.0), ("Right", -1.0)):
            arm, fore, hand = J[f"{side}Arm"], J[f"{side}ForeArm"], J[f"{side}Hand"]
            rest_dir = rest[fore] - rest[arm]
            rest_dir = np.repeat((rest_dir / np.linalg.norm(rest_dir))[None], len(R), 0)
            d = -np.cos(drop) * fwd - np.sin(drop) * np.array([0, 1.0, 0]) + 0.25 * out_sign * left
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            # the straight arm's rotation: the rest arm turned the least onto d, keeping the twist the upper arm had
            # about itself (its rest direction turned by its own rotation, then onto d)
            now = np.einsum("tab,tb->ta", R[:, arm], rest_dir)
            Rn = turn_onto(now, d) @ R[:, arm]
            old_hand_R, old_hand_P = R[:, hand].copy(), P[:, hand].copy()
            R[:, arm] = R[:, fore] = R[:, hand] = Rn
            P[:, fore] = P[:, arm] + np.einsum("tab,b->ta", Rn, rest[fore] - rest[arm])
            P[:, hand] = P[:, fore] + np.einsum("tab,b->ta", Rn, rest[hand] - rest[fore])
            D = Rn @ np.transpose(old_hand_R, (0, 2, 1))
            for j in subtree(parents, hand)[1:]:  # the fingers keep their grip
                R[:, j] = D @ R[:, j]
                P[:, j] = P[:, hand] + np.einsum("tab,tb->ta", D, P[:, j] - old_hand_P)
    unit = lambda v: v / np.linalg.norm(v)

    def part(Q, f):  # rotation Q, f of the way (about its own axis)
        ang = np.arccos(np.clip((np.trace(Q) - 1) / 2, -1, 1))
        if ang < 1e-6:
            return np.eye(3)
        ax = np.array([Q[2, 1] - Q[1, 2], Q[0, 2] - Q[2, 0], Q[1, 0] - Q[0, 1]])
        if np.linalg.norm(ax) < 1e-9:  # a half turn: the axis from the symmetric part
            ax = np.linalg.eigh(Q + Q.T)[1][:, -1]
        return axis_angle(unit(ax)[None], f * ang)[0]

    def rot(a, b):  # the least rotation taking a onto b
        ax = np.cross(a, b)
        sn = np.linalg.norm(ax)
        return np.eye(3) if sn < 1e-9 else axis_angle((ax / sn)[None], np.arctan2(sn, a @ b))[0]

    fps = float(z["fps"])
    t = np.arange(len(R)) / fps
    ss = lambda x: np.clip(x, 0, 1) ** 2 * (3 - 2 * np.clip(x, 0, 1))
    # a slice runs twice: where the wrist can't turn far enough the elbow moves the hand, which bends its path, so a
    # second pass aims at the path as it now is (one pass left the rogue's flat 30 deg into the cut, two left none)
    for mode, side, t0, t1, back_deg, out_deg, (cy, cz) in [b for b in blades for _ in range(2 if b[0] == "slice" else 1)]:
        S = "Right" if side == "R" else "Left"
        hand, fore, tip, shoulder = J[f"{S}Hand"], J[f"{S}ForeArm"], J[f"{S}HandMiddleEnd"], J[f"{S}Arm"]
        ramp = min(0.12, (t1 - t0) / 3)
        w = ss((t - t0) / ramp) * ss((t1 - t) / ramp)
        if mode == "up":
            out_dir = (-1.0 if side == "R" else 1.0) * left
            b, a = np.radians(back_deg), np.radians(out_deg)
            target = np.cos(b) * np.array([0.0, 1.0, 0.0]) - np.sin(b) * fwd  # up, tipped back
            target = np.cos(a) * target + np.sin(a) * out_dir  # and out to that side
            target /= np.linalg.norm(target, axis=1, keepdims=True)
        else:  # the hand's velocity (smoothed a little), and how sure its direction is
            vel = np.gradient(P[:, hand], axis=0) * fps
            kv = np.exp(-0.5 * (np.arange(-3, 4) / 1.0) ** 2)
            vel = np.stack([np.convolve(np.pad(vel[:, i], 3, mode="edge"), kv / kv.sum(), "valid") for i in range(3)], 1)
            w = w * np.clip(np.linalg.norm(vel, axis=1) / 0.8, 0, 1)
            kw = np.exp(-0.5 * (np.arange(-9, 10) / 3.0) ** 2)  # no sudden hand-over back to the motion's own grip
            peak = int(np.argmax(w))  # only the fade-out: easing in earlier would fight the wind-up's aim
            w[peak:] = np.minimum(np.maximum(np.convolve(np.pad(w, 9), kw / kw.sum(), "valid"), w)[peak:], w[peak])
            # each frame's target: the tip out along the arm, square to the path (r's part across v), the flat facing
            # out of the swing's plane, on the side it faced the frame before (where the plane turns over, a sign
            # picked frame by frame flipped the blade half round); then smoothed over a few frames
            live = np.flatnonzero(w > 1e-4)
            tgt = np.zeros((len(R), 3, 3))
            prev = None
            for ti in live:
                v, r = vel[ti], P[ti, hand] - P[ti, shoulder]
                yt = unit(r - (r @ v) / (v @ v) * v)
                n = unit(np.cross(yt, v))
                ref = prev if prev is not None else R[ti, hand] @ np.asarray(cz, float)
                zt = n if n @ ref > 0 else -n
                tgt[ti] = np.stack([yt, zt, np.cross(yt, zt)], 1)
                prev = zt
            kq = np.exp(-0.5 * (np.arange(-3, 4) / 0.8) ** 2)
            sm_t = tgt.copy()
            for ti in live:  # a weighted mean of the neighbours' frames, made a rotation again
                nb = [(kq[d + 3], tgt[ti + d]) for d in range(-3, 4) if 0 <= ti + d < len(R) and w[ti + d] > 1e-4]
                A = sum(k_ * T_ for k_, T_ in nb)
                U, _, Vt = np.linalg.svd(A)
                sm_t[ti] = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
            tgt = sm_t
        for ti in np.flatnonzero(w > 1e-4):
            Rh = R[ti, hand]
            y, zz = Rh @ np.asarray(cy, float), Rh @ np.asarray(cz, float)
            if mode == "up":  # only where it points
                Q = part(rot(y, target[ti]), w[ti])
            else:  # a slice: the flat square to the swing (the plane of the arm and the hand's path), the tip out along
                # the arm, so an edge leads instead of the flat (a dagger swung flat-on bats)
                Q = part(tgt[ti] @ np.stack([y, zz, np.cross(y, zz)], 1).T, w[ti])
            # the turn split between the elbow (forearm and hand together) and the wrist: the wrist bends no more than
            # wrist_max (a real wrist cocks part way; with the forearm upright, a fist-held blade can't point up from
            # the wrist alone), the elbow takes the least share that allows it
            e, h, m = P[ti, fore], P[ti, hand], P[ti, tip]
            for f in np.linspace(0, 1, 21):
                Qe = part(Q, f)
                if np.degrees(np.arccos(np.clip(unit(Qe @ (h - e)) @ unit(Q @ (m - h)), -1, 1))) <= wrist_max or f == 1:
                    break
            Qw = Q @ Qe.T
            for j in subtree(parents, fore):
                R[ti, j] = Qe @ R[ti, j]
                P[ti, j] = e + Qe @ (P[ti, j] - e)
            hp = P[ti, hand].copy()
            for j in subtree(parents, hand):
                R[ti, j] = Qw @ R[ti, j]
                P[ti, j] = hp + Qw @ (P[ti, j] - hp)
    for side, t0, t1, arc_cm in paths:
        S = "Right" if side == "R" else "Left"
        sh, el, hand = J[f"{S}Arm"], J[f"{S}ForeArm"], J[f"{S}Hand"]
        i0, i1 = int(round(t0 * fps)), min(int(round(t1 * fps)), len(R) - 1)
        a, b = P[i0, hand].copy(), P[i1, hand].copy()
        chord = b - a
        # the bow: out to that hand's side, square to the chord
        o = (-1.0 if side == "R" else 1.0) * left[(i0 + i1) // 2]
        o = unit(o - (o @ chord) / (chord @ chord) * chord)
        ramp = max(1, int(round(0.08 * fps)))
        prog = np.clip(np.einsum("ti,i->t", P[:, hand] - a, chord) / (chord @ chord), 0, 1)
        prog[i0:i1 + 1] = np.maximum.accumulate(prog[i0:i1 + 1])
        tip = J[f"{S}HandMiddleEnd"]
        frames = list(range(max(0, i0 - ramp), min(len(R), i1 + ramp + 1)))

        def solve(ti, phi):  # the arm reaching this frame's target, the elbow swung phi about the shoulder-hand line
            wt = 1.0 if i0 <= ti <= i1 else ss(1 - (i0 - ti if ti < i0 else ti - i1) / ramp)
            sp = 0.0 if ti < i0 else (1.0 if ti > i1 else prog[ti])
            H = P[ti, hand]
            T = H + wt * (a + sp * chord + arc_cm / 100 * np.sin(np.pi * sp) * o - H)
            Sp, E = P[ti, sh], P[ti, el]
            L1, L2 = np.linalg.norm(E - Sp), np.linalg.norm(H - E)
            d = min(np.linalg.norm(T - Sp), (L1 + L2) * 0.999)
            u = unit(T - Sp)
            pole = (E - Sp) - ((E - Sp) @ u) * u
            pole = unit(pole) if np.linalg.norm(pole) > 1e-6 else unit(np.cross(u, left[ti]))
            pole = axis_angle(u[None], phi)[0] @ pole
            ad = (L1 ** 2 - L2 ** 2 + d ** 2) / (2 * d)
            return Sp + ad * u + np.sqrt(max(L1 ** 2 - ad ** 2, 0)) * pole, Sp + d * u

        # the elbow's swing: the hand keeps its turn (the blade its angle), so pick, frame by frame, where round the
        # shoulder-hand line the elbow bends the wrist least (it can sit anywhere without moving the hand), preferring
        # its own place; then smoothed. Keeping the elbow where it was bent the rogue's wrist to 114 deg
        phis = np.radians(np.arange(-90, 91, 5))
        best = []
        for ti in frames:
            hd = unit(P[ti, tip] - P[ti, hand])
            cost = []
            for ph in phis:
                E2, T = solve(ti, ph)
                bend = np.degrees(np.arccos(np.clip(unit(T - E2) @ hd, -1, 1)))
                cost.append(max(bend - 30, 0) ** 2 + 0.02 * np.degrees(ph) ** 2)
            best.append(phis[int(np.argmin(cost))])
        best = np.array(best)
        kb = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)
        best = np.convolve(np.pad(best, 3, mode="edge"), kb / kb.sum(), "valid")
        for ti, ph in zip(frames, best):
            H, E, Sp = P[ti, hand].copy(), P[ti, el].copy(), P[ti, sh].copy()
            E2, T = solve(ti, ph)
            Q1 = rot(unit(E - Sp), unit(E2 - Sp))
            Q2 = rot(unit(Q1 @ (H - E)), unit(T - E2))
            R[ti, sh] = Q1 @ R[ti, sh]
            R[ti, el] = Q2 @ Q1 @ R[ti, el]
            P[ti, el] = E2
            for j in subtree(parents, hand):  # the hand keeps its turn (and the blade its angle); it only moves
                P[ti, j] = P[ti, j] + (T - H)
        # where the arm straightens the elbow's swing can't help: the wrist eased back to wrist_max + 5 deg (the blade
        # gives a little there, after the cut), the correction smoothed over a few frames so the blade doesn't flick
        rv = np.zeros((len(R), 3))
        for ti in frames:
            f_, h_ = unit(P[ti, hand] - P[ti, el]), unit(P[ti, tip] - P[ti, hand])
            bend = np.degrees(np.arccos(np.clip(f_ @ h_, -1, 1)))
            if bend > wrist_max + 5:
                rv[ti] = unit(np.cross(h_, f_)) * np.radians(bend - wrist_max - 5)
        need = rv
        rv = np.stack([np.convolve(np.pad(rv[:, i], 3, mode="edge"), kb / kb.sum(), "valid") for i in range(3)], 1)
        short = np.linalg.norm(rv, axis=1) < np.linalg.norm(need, axis=1)  # smoothing eases it in and out, never short
        rv[short] = need[short]
        for ti in np.flatnonzero(np.linalg.norm(rv, axis=1) > 1e-4):
            Qc = axis_angle(unit(rv[ti])[None], np.linalg.norm(rv[ti]))[0]
            hp = P[ti, hand].copy()
            for j in subtree(parents, hand):
                R[ti, j] = Qc @ R[ti, j]
                P[ti, j] = hp + Qc @ (P[ti, j] - hp)
    out = dict(z)
    out["rots"], out["joints"] = R, P
    return out


def blade_in_hand(z, rig, side):
    """Where what a hand holds points (its socket's +Y) and faces (+Z) in the frame of Kimodo's hand joint: the retarget turns
    SOMA's rest hand (wrist to middle fingertip) the least onto the rig's hand bone (retarget_kimodo.py's S), so at
    SOMA's rest the socket points S^T y, in Kimodo's axes M^T S^T y."""
    txt = subprocess.run([BLENDER, "-b", rig / "rigged.blend", "-P", ROOT / "rigging/blender/socket_axes.py"],
                         capture_output=True, text=True, check=True).stdout
    axes = json.loads(next(l for l in txt.splitlines() if l.startswith("SOCKET_AXES"))[len("SOCKET_AXES"):])[side]
    tmpl = json.loads((rig / "rig_info.json").read_text()).get("template", "humanoid")
    aim = json.loads((ROOT / "rigging/templates" / f"{tmpl}_kimodo.json").read_text())["aim"][f"hand_{side}"]
    names = [str(n) for n in z["names"]]
    M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # Kimodo's axes -> Blender's
    rb = z["rest"] @ M.T
    u = rb[names.index(aim[1])] - rb[names.index(aim[0])]
    S = turn_onto((u / np.linalg.norm(u))[None], (np.array(axes["hand_y"]) / np.linalg.norm(axes["hand_y"]))[None])[0]
    cy, cz = (M.T @ S.T @ np.array(axes[k]) for k in ("socket_y", "socket_z"))
    return cy / np.linalg.norm(cy), cz / np.linalg.norm(cz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("capture")
    ap.add_argument("--name", required=True)
    ap.add_argument("--lean", type=float, default=0.0, help="degrees forward (+) or back (-) at the waist")
    ap.add_argument("--arms-back", type=float, help="arms straight back, this many degrees below horizontal")
    ap.add_argument("--head", type=float, default=0.5, help="the neck turned back by this share of the lean")
    ap.add_argument("--tuck", type=float, default=0.0, help="upper arms turned in toward the body by this many degrees")
    ap.add_argument("--pump", type=float, help="arms rebuilt as a runner's: swing this times the opposite thigh's")
    ap.add_argument("--elbow", type=float, default=90.0, help="with --pump: the elbow's inner angle")
    ap.add_argument("--elbow-out", type=float, default=8.0, help="with --pump: elbows this many degrees out from the body")
    ap.add_argument("--pump-forward", type=float, default=10.0, help="with --pump: the swing's centre, ahead of hanging")
    ap.add_argument("--feet-forward", action="store_true", help="knees aimed ahead, feet in line under them")
    ap.add_argument("--blade-up", action="append", default=[], help="SIDE:FROM-TO[:BACK_DEG[:OUT_DEG]]: the held item points up, tipped back")
    ap.add_argument("--blade-slice", action="append", default=[], help="SIDE:FROM-TO: the held blade cuts edge first")
    ap.add_argument("--strike-path", action="append", default=[], help="SIDE:FROM-TO[:ARC_CM]: the hand on a near-straight line")
    ap.add_argument("--move")
    a = ap.parse_args()
    p = Path(a.project).resolve()
    clips, rig = p / "anymove/runs/clips", p / "rigging/rigged"
    src = Path(a.capture)
    src = src.resolve() if src.is_dir() else clips / a.capture
    z = dict(np.load(src / "kimodo.npz"))
    blades = []
    for b in a.blade_up:
        side, span, *deg = b.split(":")
        t0, t1 = map(float, span.split("-"))
        back, out_deg = (list(map(float, deg)) + [45.0, 0.0][len(deg):])[:2]
        blades.append(("up", side.upper(), t0, t1, back, out_deg, blade_in_hand(z, rig, side.upper())))
    for b in a.blade_slice:
        side, span = b.split(":")
        t0, t1 = map(float, span.split("-"))
        blades.append(("slice", side.upper(), t0, t1, 0, 0, blade_in_hand(z, rig, side.upper())))
    paths = []
    for b in a.strike_path:
        side, span, *arc = b.split(":")
        t0, t1 = map(float, span.split("-"))
        paths.append((side.upper(), t0, t1, float(arc[0]) if arc else 4.0))
    new = posture(z, a.lean, a.arms_back, a.head, a.tuck, a.pump, a.elbow, a.elbow_out, a.pump_forward, a.feet_forward,
                  blades, paths=paths)
    what = ", ".join(filter(None, [f"leaning {a.lean:+g} deg" if a.lean else "",
                                   f"arms straight back {a.arms_back:g} deg below horizontal" if a.arms_back is not None else "",
                                   f"elbows tucked {a.tuck:g} deg" if a.tuck else "",
                                   f"arms pumping {a.pump:g}x the legs, elbows {a.elbow:g} deg, {a.elbow_out:g} deg out" if a.pump is not None else "",
                                   "feet forward" if a.feet_forward else "",
                                   *[f"blade up ({b})" for b in a.blade_up],
                                   *[f"blade slicing ({b})" for b in a.blade_slice],
                                   *[f"strike path ({b})" for b in a.strike_path]]))
    new["text"] = np.array(f"{z['text']} ({what})")
    out = clips / a.name
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    np.savez(out / "kimodo.npz", **new)
    tmpl = json.loads((rig / "rig_info.json").read_text()).get("template", "humanoid")
    mapping = ROOT / "rigging/templates" / f"{tmpl}_kimodo.json"
    run([BLENDER, "-b", rig / "rigged.blend", "-P", ROOT / "rigging/blender/retarget_kimodo.py", "--", mapping,
         out / "kimodo.npz", rig / "rig_info.json", out / "motion.json"], cwd=ROOT, stdout=subprocess.DEVNULL)
    springs = ["--springs", rig / "springs.json"] if (rig / "springs.json").exists() else []
    with open(out / "blender.log", "w") as log:
        run([BLENDER, "-b", "-P", "blender/apply_motion.py", "--", rig / "rigged.blend", rig / "rig_info.json",
             out / "motion.json", out / "render", "--glb", *springs], cwd=ROOT / "external/anymove", stdout=log,
            stderr=subprocess.STDOUT)
    src_info = json.loads((src / "source.json").read_text()) if (src / "source.json").exists() else {}
    built = time.strftime("%Y-%m-%d %H:%M", time.localtime((rig / "rigged.blend").stat().st_mtime))
    made_on = json.loads((rig.parent / "promote.json").read_text()).get("model") if (rig.parent / "promote.json").exists() else None
    (out / "source.json").write_text(json.dumps({
        "move": a.move or src_info.get("move", src.name), "kimodo": str(new["text"]), "speed": 1,
        "posture": {"from": str(src), "lean_deg": a.lean, "arms_back_deg": a.arms_back, "head": a.head, "tuck_deg": a.tuck,
                    "pump": a.pump, "elbow_deg": a.elbow, "elbow_out_deg": a.elbow_out, "pump_forward_deg": a.pump_forward,
                    "feet_forward": a.feet_forward, "blade_up": a.blade_up,
                    "blade_slice": a.blade_slice, "strike_path": a.strike_path},
        "rig": {"model": made_on, "built": built}}, indent=1))
    print(f"posture: {what or 'unchanged'} -> capture {out}", flush=True)


if __name__ == "__main__":
    main()
