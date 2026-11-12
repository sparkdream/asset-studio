"""A single Kimodo pose made by hand, as a capture to pin Kimodo to (rigging/kimodo_capture.py --pose / recipes): for a
pose no prompt reaches (Kimodo's "slide" never took the hips below 0.78 m).

    python -m rigging.kimodo_pose PROJECT_DIR SPEC.json --name NAME

SPEC.json:
    {"from": "CAPTURE:FRAME",            the pose it starts from (arms, hands, head as there)
     "hips_height": 0.30,                the hips above the ground (metres), facing +z on the start's spot
     "lean_back": 40, "head": 0.5,       the upper body tipped back (degrees) at the hips; the neck turned back by
                                         `head` of it, so the head looks ahead
     "legs": {"Left":  {"thigh": [0, -0.12, 0.99], "knee": 0},
              "Right": {"thigh": [-0.57, -0.1, 0.82], "knee": 110}}}
A leg's "thigh" is the hip-to-knee direction in the character's axes (x her left, y up, z forward); "knee" its bend
(degrees from straight), "shin" an optional direction for the shin that picks the bend's plane, "ankle" the toes
pointed by that many degrees (a kneeling leg's foot lies on its instep). The knee bends as a hinge: the shin swings back from the thigh in the thigh's own plane, and a
straight leg's kneecap faces where it would if the leg had swung there from hanging. Without "ankle" the foot keeps the
shin's turn (ankle neutral: toes up on a leg held out in front). Positions follow from SOMA's rest bone offsets.

Writes PROJECT_DIR/anymove/runs/clips/NAME/kimodo.npz (one frame) and source.json; not retargeted (a pin, not a move).
"""
import argparse, json
from pathlib import Path

import numpy as np

from rigging.kimodo_posture import axis_angle, subtree, turn_onto


def unit(v):
    return v / np.linalg.norm(v)


def build(z, f, spec):
    names = [str(n) for n in z["names"]]
    J = {n: i for i, n in enumerate(names)}
    parents, rest = z["parents"], z["rest"]
    R = z["rots"][f].astype(np.float64).copy()
    # face +z: turn the whole pose about y so the hips face forward
    fh = R[J["Hips"], :, 2]
    yaw = np.arctan2(fh[0], fh[2])
    Y = axis_angle(np.array([[0.0, 1.0, 0.0]]), -yaw)[0]
    R = np.einsum("ab,jbc->jac", Y, R)
    X = lambda deg: axis_angle(np.array([[1.0, 0.0, 0.0]]), np.radians(deg))[0]  # about her left: + tips up forward
    lean = float(spec.get("lean_back", 0))
    if lean:  # the upper body (and arms) tipped back at the hips, the head brought back toward upright
        for j in subtree(parents, J["Spine1"]):
            R[j] = X(-lean) @ R[j]
        for j in subtree(parents, J["Neck1"]):
            R[j] = X(lean * float(spec.get("head", 0.5))) @ R[j]
    down = np.array([0.0, -1.0, 0.0])
    for side, leg in spec.get("legs", {}).items():
        hip, knee, foot = J[f"{side}Leg"], J[f"{side}Shin"], J[f"{side}Foot"]
        t = unit(np.array(leg["thigh"], float))
        th = np.radians(float(leg.get("knee", 0)))
        if th < np.radians(1):  # straight: the kneecap where swinging from hanging puts it
            k = turn_onto(down[None], t[None])[0] @ np.array([0.0, 0.0, 1.0])
            th = 0.0
        else:  # the shin swings back from the thigh by th: shin = cos th t - sin th k, k the kneecap's way
            hint = np.array(leg.get("shin", [0.0, 0.0, 0.0]), float)
            k = turn_onto(down[None], t[None])[0] @ np.array([0.0, 0.0, 1.0])
            if hint.any():  # an explicit shin direction picks the bend's plane
                s_ = unit(hint)
                k = unit(np.cos(th) * t - s_)
        k = unit(k - (k @ t) * t)
        Rt = np.stack([np.cross(-t, k), -t, k], 1)  # columns: x (her left at rest), y (up the thigh), z (kneecap)
        Rs = Rt @ X(np.degrees(th))  # the knee: a hinge about the thigh's x
        old_thigh = R[hip].copy()
        D_th = Rt @ old_thigh.T
        for j in subtree(parents, hip):  # the whole leg turned with the thigh, then the shin and foot set
            R[j] = D_th @ R[j]
        D_sh = Rs @ R[knee].T
        for j in subtree(parents, knee):
            R[j] = D_sh @ R[j]
        D_ft = Rs @ X(-float(leg.get("ankle", 0))) @ R[foot].T  # "ankle": the toes pointed (degrees), else neutral
        for j in subtree(parents, foot):
            R[j] = D_ft @ R[j]
    # positions from the rest offsets, the hips at the asked height on the start's spot
    P = np.zeros((len(names), 3))
    order = subtree(parents, J["Hips"])  # breadth first from the root: parents before children
    P[J["Hips"]] = [0.0, float(spec.get("hips_height", rest[J["Hips"], 1])), 0.0]
    for j in order:
        p = int(parents[j])
        if p >= 0:
            P[j] = P[p] + R[p] @ (rest[j] - rest[p])
    return R, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("spec")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    p = Path(a.project).resolve()
    clips = p / "anymove/runs/clips"
    spec = json.loads(Path(a.spec).read_text())
    src, _, fr = spec["from"].partition(":")
    z = dict(np.load(clips / src / "kimodo.npz"))
    R, P = build(z, int(fr) if fr else 0, spec)
    out = clips / a.name
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    keep = {k: z[k] for k in ("rest", "names", "parents", "fps")}
    np.savez(out / "kimodo.npz", rots=R[None].astype(np.float32), joints=P[None].astype(np.float32),
             foot_contacts=np.zeros((1, z["foot_contacts"].shape[1]), bool), text=np.array("a pose made by hand"), **keep)
    (out / "source.json").write_text(json.dumps({"move": "pose", "pose": spec, "kimodo": "a pose made by hand"}, indent=1))
    print(f"pose -> {out}")


if __name__ == "__main__":
    main()
