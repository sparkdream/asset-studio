"""Two Kimodo captures of a project played one after the other, as one capture: a move, then what follows it (a
strike that lands in a lunge, then a stand-up back to the idle stance), without making the first one again.

    python -m rigging.join_moves PROJECT_DIR FIRST SECOND [--name NAME] [--move MOVE] [--blend 0.2]

SECOND should start where FIRST ends: make it with Kimodo pinned to FIRST's last frame (rigging/kimodo_capture.py
--pose 0=FIRST_DIR:LAST_FRAME). Kimodo puts every motion where it likes, so SECOND is turned about the vertical and
moved across the ground until its first frame stands where FIRST's last one does (its height is kept), and its first
frame is dropped. What still differs at the seam (a pin is close, not exact) is faded out over --blend seconds of
SECOND: each joint's rotation and the hips' position start as FIRST's last frame and ease into SECOND's own.

The joined Kimodo motion is retargeted onto the project's rig (rigging/blender/retarget_kimodo.py, which also grounds
the feet) and exported like any capture: PROJECT_DIR/anymove/runs/clips/NAME/ (default FIRST_then_SECOND's move) with
kimodo.npz, motion.json, render/animation.glb and source.json (the move is FIRST's unless --move).
"""
import argparse, json, subprocess, time
from pathlib import Path

import numpy as np

from rigging.kimodo_capture import BLENDER, ROOT, run


def yaw_of(R):
    """The heading (radians about +y, Kimodo's up) of rotations R (..., 3, 3): where they turn +z (forward)."""
    f = R[..., :, 2]
    return np.arctan2(f[..., 0], f[..., 2])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def slerp_rots(R0, R1, w):
    """Per joint, rotation R0 eased toward R1 by weight w (0: R0, 1: R1): R0 (J,3,3), R1 (J,3,3)."""
    D = np.einsum("jba,jbc->jac", R0, R1)  # R0^T R1
    ang = np.arccos(np.clip((np.trace(D, axis1=1, axis2=2) - 1) / 2, -1, 1))
    axis = np.stack([D[:, 2, 1] - D[:, 1, 2], D[:, 0, 2] - D[:, 2, 0], D[:, 1, 0] - D[:, 0, 1]], -1)
    axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-9)
    a = ang * w
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = axis[:, 2], -axis[:, 1], axis[:, 0]
    S = np.eye(3) + np.sin(a)[:, None, None] * K + (1 - np.cos(a))[:, None, None] * K @ K
    return R0 @ S


def join(a, b, blend_frames):
    """The two Kimodo motions (dicts of arrays) as one (see the module docstring)."""
    names = [str(n) for n in a["names"]]
    hips = names.index("Hips")
    turn = yaw_of(a["rots"][-1, hips]) - yaw_of(b["rots"][0, hips])
    Ry = rot_y(turn)
    rots = np.einsum("ab,tjbc->tjac", Ry, b["rots"])
    joints = np.einsum("ab,tjb->tja", Ry, b["joints"])
    shift = a["joints"][-1, hips] - joints[0, hips]
    shift[1] = 0  # height as Kimodo made it
    joints = joints + shift
    rots, joints = rots[1:], joints[1:]
    contacts = b["foot_contacts"][1:]
    # the seam: what differs between FIRST's last frame and SECOND's first fades out (smoothstep over blend_frames)
    gap_rot, gap_pos = a["rots"][-1], a["joints"][-1] - (np.einsum("ab,jb->ja", Ry, b["joints"][0]) + shift)
    for t in range(min(blend_frames, len(rots))):
        x = (t + 1) / (blend_frames + 1)
        w = x * x * (3 - 2 * x)
        start = np.einsum("jab,jcb->jac", gap_rot, np.einsum("ab,jbc->jac", Ry, b["rots"][0]))  # A_last B_first^T
        rots[t] = slerp_rots(np.einsum("jab,jbc->jac", start, rots[t]), rots[t], w)
        joints[t] = joints[t] + (1 - w) * gap_pos
    seam = float(np.degrees(np.arccos(np.clip((np.trace(np.einsum("jba,jbc->jac", a["rots"][-1],
                 np.einsum("ab,jbc->jac", Ry, b["rots"][0])), axis1=1, axis2=2) - 1) / 2, -1, 1))).max())
    out = {k: a[k] for k in a}
    out["rots"] = np.concatenate([a["rots"], rots])
    out["joints"] = np.concatenate([a["joints"], joints])
    out["foot_contacts"] = np.concatenate([a["foot_contacts"], contacts])
    out["text"] = np.array(f"{a['text']} Then: {b['text']}")
    return out, seam, float(np.degrees(turn)), float(np.linalg.norm(gap_pos[hips]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("--name")
    ap.add_argument("--move")
    ap.add_argument("--blend", type=float, default=0.2, help="seconds of SECOND over which the seam fades out")
    a = ap.parse_args()
    p = Path(a.project).resolve()
    clips, rig = p / "anymove/runs/clips", p / "rigging/rigged"
    A, B = (dict(np.load(clips / n / "kimodo.npz")) for n in (a.first, a.second))
    fps = float(A["fps"])
    if float(B["fps"]) != fps:
        raise SystemExit("the two motions have different frame rates")
    src_a = json.loads((clips / a.first / "source.json").read_text())
    src_b = json.loads((clips / a.second / "source.json").read_text())
    joined, seam, turn, gap = join(A, B, int(round(a.blend * fps)))
    name = a.name or f"{a.first}_then_{src_b.get('move', a.second)}"
    out = clips / name
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    np.savez(out / "kimodo.npz", **joined)
    print(f"joined: {len(A['rots'])} + {len(B['rots']) - 1} frames; SECOND turned {turn:.1f} deg onto FIRST's end; "
          f"at the seam the largest joint turn {seam:.1f} deg and the hips {gap * 100:.1f} cm apart (faded over "
          f"{a.blend:.2f} s)", flush=True)
    tmpl = json.loads((rig / "rig_info.json").read_text()).get("template", "humanoid")
    mapping = ROOT / "rigging/templates" / f"{tmpl}_kimodo.json"
    run([BLENDER, "-b", rig / "rigged.blend", "-P", ROOT / "rigging/blender/retarget_kimodo.py", "--", mapping,
         out / "kimodo.npz", rig / "rig_info.json", out / "motion.json"], cwd=ROOT, stdout=subprocess.DEVNULL)
    springs = ["--springs", rig / "springs.json"] if (rig / "springs.json").exists() else []
    with open(out / "blender.log", "w") as log:
        run([BLENDER, "-b", "-P", "blender/apply_motion.py", "--", rig / "rigged.blend", rig / "rig_info.json",
             out / "motion.json", out / "render", "--glb", *springs], cwd=ROOT / "external/anymove", stdout=log,
            stderr=subprocess.STDOUT)
    built = time.strftime("%Y-%m-%d %H:%M", time.localtime((rig / "rigged.blend").stat().st_mtime))
    made_on = json.loads((rig.parent / "promote.json").read_text()).get("model") if (rig.parent / "promote.json").exists() else None
    (out / "source.json").write_text(json.dumps({
        "move": a.move or src_a.get("move", a.first), "kimodo": str(joined["text"]), "speed": 1,
        "joined": {"first": a.first, "second": a.second, "blend_s": a.blend, "seam_deg": round(seam, 1),
                   "seam_hips_cm": round(gap * 100, 1)},
        "rig": {"model": made_on, "built": built}}, indent=1))
    print(f"capture {out}", flush=True)


if __name__ == "__main__":
    main()
