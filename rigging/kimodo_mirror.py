"""A Kimodo capture mirrored left for right, as a new capture: a strike with the right hand becomes the same strike
with the left, exactly (Kimodo asked for the other hand makes a different move).

    python -m rigging.kimodo_mirror PROJECT_DIR CAPTURE --name NAME [--move MOVE]

CAPTURE is a capture folder holding kimodo.npz (in PROJECT_DIR's anymove/runs/clips, or a path to one). Each Left*
joint takes its Right* twin's motion and the other way round, reflected across the character's mid plane (Kimodo's
x, the character's left, is negated): positions P'_j = M P_twin, world turns R'_j = M R_twin M with M = diag(-1, 1, 1),
which keep being rotations, and SOMA's rest pose is symmetric so they apply to the twin's rest bone. The foot contacts
swap halves. The motion is retargeted onto the project's rig and exported like any capture.
"""
import argparse, json, subprocess, time
from pathlib import Path

import numpy as np

from rigging.kimodo_capture import BLENDER, ROOT, run


def twin(name):
    for a, b in (("Left", "Right"), ("Right", "Left")):
        if name.startswith(a):
            return b + name[len(a):]
    return name


def mirror(z):
    names = [str(n) for n in z["names"]]
    J = {n: i for i, n in enumerate(names)}
    tw = np.array([J.get(twin(n), i) for i, n in enumerate(names)])
    M = np.diag([-1.0, 1.0, 1.0])
    out = {k: z[k] for k in z}
    out["joints"] = np.einsum("ab,tjb->tja", M, z["joints"][:, tw]).astype(z["joints"].dtype)
    out["rots"] = np.einsum("ab,tjbc,cd->tjad", M, z["rots"][:, tw], M).astype(z["rots"].dtype)
    fc = z["foot_contacts"]
    h = fc.shape[1] // 2
    out["foot_contacts"] = np.concatenate([fc[:, h:], fc[:, :h]], 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("capture")
    ap.add_argument("--name", required=True)
    ap.add_argument("--move")
    a = ap.parse_args()
    p = Path(a.project).resolve()
    clips, rig = p / "anymove/runs/clips", p / "rigging/rigged"
    src = Path(a.capture)
    src = src.resolve() if src.is_dir() else clips / a.capture
    z = dict(np.load(src / "kimodo.npz"))
    names = [str(n) for n in z["names"]]
    J = {n: i for i, n in enumerate(names)}
    rest = z["rest"]
    err = max((np.abs(rest[J[twin(n)]] * [-1, 1, 1] - rest[i]).max() for i, n in enumerate(names) if twin(n) in J),
              default=0)
    if err > 0.01:
        raise SystemExit(f"the rest pose isn't symmetric ({err * 100:.1f} cm): can't mirror")
    new = mirror(z)
    new["text"] = np.array(f"{z['text']} (mirrored)")
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
        "mirrored": {"from": str(src)}, "rig": {"model": made_on, "built": built}}, indent=1))
    print(f"mirrored -> capture {out}", flush=True)


if __name__ == "__main__":
    main()
