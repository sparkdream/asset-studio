"""Make a capture loop: find the stretch of its motion whose end matches its start, close the seam, and export it as a
capture of its own (a walk cycle from a walk, an idle from a stance...). Works on any capture's motion.json, whatever made
it (a clip read by SAM, Kimodo) and whatever the rig.

    python -m rigging.loop CAPTURE_DIR [--name NAME] [--in-place] [--min 0.6] [--max 3.0] [--skip 0.3]
    python -m rigging.loop LOOP_DIR --fix-height      # a loop made before loops kept their height (see fix_height)

1. The loop: of every stretch from --min to --max seconds, not starting in the first --skip seconds (a walk starts from
   standing), the one whose last frame best matches its first, in pose and in how the pose is moving (the bones' angles
   and their velocities: a foot on its way forward and on its way back look alike in one frame).
2. The seam: what still differs between the end and the start is spread over the loop, so its end meets its start
   exactly: each bone's angles (unwrapped), the root's turn and height, and its sideways sway.
3. The travel: by default the root keeps moving (root motion: the game moves the character by the travel per cycle,
   saved in source.json); --in-place takes the forward travel out and leaves the sway.
Writes CAPTURE_DIR/../NAME/ (default <capture>_loop) with motion.json, source.json and render/animation.glb, on the rig
the capture was made on (the project's rigging/rigged/, else anymove's rigged/).
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BLENDER = os.environ.get("BLENDER", "blender")


def qmul(a, b):
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qpow(q, f):
    """q (unit, w >= 0) raised to the power f: the same turn about the same axis, f times as far."""
    q = q * np.sign(q[..., :1] + 1e-12)
    ang = 2 * np.arccos(np.clip(q[..., 0], -1, 1))
    axis = q[..., 1:] / np.maximum(np.linalg.norm(q[..., 1:], axis=-1, keepdims=True), 1e-12)
    half = ang * f / 2
    return np.concatenate([np.cos(half)[..., None], axis * np.sin(half)[..., None]], -1)


def find_loop(E, fps, lo, hi, skip):
    """(a, b, cost): frames a < b with b's pose and motion nearest a's; the loop plays a .. b-1."""
    feat = np.concatenate([np.sin(E), np.cos(E)], 1)
    vel = np.gradient(E, axis=0) * fps
    sv = vel.std(0) + 1e-6  # each channel's own scale, so a stiff spine counts as much as a swinging arm
    best = (None, None, np.inf)
    T = len(E)
    for a in range(int(skip * fps), T):
        for b in range(a + int(lo * fps), min(T, a + int(hi * fps) + 1)):
            cost = np.sum((feat[b] - feat[a]) ** 2) + 0.05 * np.sum(((vel[b] - vel[a]) / sv) ** 2) / len(sv)
            if cost < best[2]:
                best = (a, b, cost)
    return best


def export(out):
    """Render a capture's motion.json on the rig it was made on: render/animation.glb."""
    c = out.parent.parent.parent  # the creature folder (anymove/): runs/clips/<capture>
    ch = c.parent / "rigging"
    rig = ch / "rigged" if (ch / "character.json").exists() else c / "rigged"
    springs = ["--springs", rig / "springs.json"] if (rig / "springs.json").exists() else []
    with open(out / "blender.log", "w") as log:
        subprocess.run(list(map(str, [BLENDER, "-b", "-P", "blender/apply_motion.py", "--", rig / "rigged.blend",
                                      rig / "rig_info.json", out / "motion.json", out / "render", "--glb", *springs])),
                       cwd=ROOT / "external/anymove", stdout=log, stderr=subprocess.STDOUT, check=True)
    return rig


def fix_height(out):
    """A loop made before its height was kept (see main): lowered so its lowest foot over the cycle sits as high as the
    rig's lowest foot at rest, as the retarget grounded the capture it came from. The height it lost was one constant
    (the capture's at the loop's first frame), so one constant puts it back; a run's flight keeps its air."""
    m = json.loads((out / "motion.json").read_text())
    src = json.loads((out / "source.json").read_text()) if (out / "source.json").exists() else {}
    if not m.get("loop") or "joints" not in m:
        sys.exit(f"{out.name}: not a loop with its joints (nothing to measure the feet by)")
    if src.get("loop", {}).get("height") == "kept":
        print(f"{out.name}: made with its height kept: nothing to fix")
        return
    c = out.parent.parent.parent
    ch = c.parent / "rigging"
    info = json.loads(((ch / "rigged" if (ch / "character.json").exists() else c / "rigged") / "rig_info.json").read_text())
    tmpl = json.loads((ROOT / "rigging/templates" / f"{info.get('template', 'humanoid')}_kimodo.json").read_text())
    is_foot = lambda n: n.split("_")[0] in tmpl.get("feet", [])
    rest = min(k["rest"][2] for k in info["keypoints"] if is_foot(k["name"]))
    J = np.array(m["joints"], float)
    feet = [i for i, n in enumerate(m["joint_names"]) if is_foot(n)]
    shift = rest - J[:, feet, 2].min()
    tr = np.array(m["trans"], float)
    tr[:, 2] += shift
    J[:, :, 2] += shift
    m["trans"], m["joints"] = tr.round(5).tolist(), J.round(5).tolist()
    (out / "motion.json").write_text(json.dumps(m))
    src.setdefault("loop", {})["height"] = "kept"
    src["loop"]["height_fixed_m"] = round(float(shift), 4)
    (out / "source.json").write_text(json.dumps(src, indent=1))
    export(out)
    print(f"{out.name}: lowered {-shift:.3f} m, exported again")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--fix-height", action="store_true", help="CAPTURE is a loop made before loops kept their height: "
                    "put its feet back on the ground (fix_height) and export it again")
    ap.add_argument("--name")
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--min", type=float, default=0.6)
    ap.add_argument("--max", type=float, default=3.0)
    ap.add_argument("--skip", type=float, default=0.3)
    a = ap.parse_args()
    k = Path(a.capture).resolve()
    if a.fix_height:
        return fix_height(k)
    m = json.loads((k / "motion.json").read_text())
    fps = float(m.get("fps", 24))
    E = np.unwrap(np.array(m["euler"], float).reshape(len(m["euler"]), -1), axis=0)
    T = len(E)
    if T < int((a.skip + a.min) * fps) + 2:
        sys.exit(f"{T} frames is too short for a loop of at least {a.min} s after {a.skip} s")
    s, e, cost = find_loop(E, fps, a.min, a.max, a.skip)
    n = e - s
    f = np.arange(n + 1)[:, None] / n  # 0 at the loop's start, 1 at its end (frame e, which becomes the start again)

    # the seam: spread the end's difference from the start over the loop, so frame e lands exactly on frame s
    Es = E[s:e + 1] - f * (E[e] - E[s])
    Q = np.array(m["root_quat"], float)[s:e + 1]
    Q *= np.sign(np.sum(Q * Q[:1], 1, keepdims=True))  # one hemisphere
    err = qmul(Q[-1:], Q[:1] * np.array([1, -1, -1, -1]))[0]  # the turn from the start's root to the end's
    Qs = qmul(qpow(np.repeat(err[None] * np.array([1, -1, -1, -1]), n + 1, 0), f[:, 0]), Q)
    tr = np.array(m["trans"], float)[s:e + 1]
    # across the ground the loop starts where the character stands; its height stays as captured (it holds a crouch,
    # and the retarget's grounding of the feet). Taking the start's height off too stood every loop up by it: the
    # rogue's crouch walk floated 0.3 m, her run 0.1 m.
    tr[:, :2] -= tr[0, :2]
    d = tr[-1] - tr[0]
    travel = d.copy()
    travel[2] = 0  # height: the seam closes it; horizontal: the travel per cycle
    if a.in_place:  # the forward progression out, the sway kept
        tr = tr - f * d
        travel[:] = 0
    else:
        tr[:, 2] -= f[:, 0] * d[2]
    loop = {**m, "euler": Es[:-1].reshape(n, len(m["bones"]), 3).round(5).tolist(), "root_quat": Qs[:-1].round(6).tolist(),
            "trans": tr[:-1].round(5).tolist()}
    if "joints" in m:  # the joints' positions follow the root's travel; the seam's small changes are left out
        J = np.array(m["joints"], float)[s:e]
        J = J - np.array(m["trans"], float)[s:e, None] + tr[:-1, None]
        loop["joints"] = J.round(5).tolist()
    loop["loop"] = True

    out = k.parent / (a.name or f"{k.name}_loop{'_in_place' if a.in_place else ''}")
    (out / "render").mkdir(parents=True, exist_ok=True)
    (out / "motion.json").write_text(json.dumps(loop))
    src = json.loads((k / "source.json").read_text()) if (k / "source.json").exists() else {}
    seam = float(np.degrees(np.abs(E[e] - E[s]).max()))
    yaw = float(np.degrees(2 * np.arccos(min(1.0, abs(err[0])))))
    info = {"from": k.name, "frames": [int(s), int(e)], "seconds": round(n / fps, 3), "in_place": a.in_place,
            "travel_per_cycle_m": travel.round(4).tolist(), "seam_max_deg": round(seam, 1), "root_turn_closed_deg": round(yaw, 1),
            "height": "kept"}
    (out / "source.json").write_text(json.dumps({**{x: src[x] for x in ("move", "kimodo", "rig") if x in src},
                                                 "speed": 1, "loop": info}, indent=1))
    print(f"loop: frames {s}..{e} ({n / fps:.2f} s), the largest angle closed at the seam {seam:.1f} deg, root turn "
          f"{yaw:.1f} deg, travel per cycle {np.linalg.norm(travel):.2f} m", flush=True)

    export(out)
    print(f"capture {out}")


if __name__ == "__main__":
    main()
