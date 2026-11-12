"""Another character's captures made again on this project's rig.

    python -m rigging.copy_moves SOURCE_PROJECT_DIR PROJECT_DIR CAPTURE [CAPTURE ...]

A capture's motion.json turns its own rig's bones from that rig's rest pose, and every rig rests differently (each is
fitted to its mesh), so it can't be copied as it is:
- a Kimodo capture still has Kimodo's own motion (kimodo.npz): it is retargeted onto this rig
  (rigging/blender/retarget_kimodo.py), as rigging/kimodo_capture.py does with a new one;
- any other (a loop, a capture read from a video) is retargeted from the other rig's motion
  (rigging/blender/retarget_rig.py): each bone turns in the world as the other rig's does, each limb points where the
  other's points.
Then exported on this rig. Both rigs must come from the same template. Captures this project already has are kept.
source.json is the original's, with this rig and "copied_from".
"""
import json, shutil, subprocess, sys, time
from pathlib import Path

from rigging.kimodo_capture import BLENDER, ROOT, run


def copy(src: Path, dst: Path, name: str):
    s, d = src / "anymove/runs/clips" / name, dst / "anymove/runs/clips" / name
    if d.exists():
        print(f"{name}: already here", flush=True)
        return
    srig, rig = src / "rigging/rigged", dst / "rigging/rigged"
    tmpl = json.loads((rig / "rig_info.json").read_text()).get("template", "humanoid")
    if json.loads((srig / "rig_info.json").read_text()).get("template", "humanoid") != tmpl:
        print(f"{name}: that rig is from another template than this one's ({tmpl})", flush=True)
        return
    mapping = ROOT / "rigging/templates" / f"{tmpl}_kimodo.json"
    source = json.loads((s / "source.json").read_text()) if (s / "source.json").exists() else {}
    d.mkdir(parents=True)
    if (s / "kimodo.npz").exists():
        shutil.copy(s / "kimodo.npz", d / "kimodo.npz")
        speed = source.get("speed", 1)
        run([BLENDER, "-b", rig / "rigged.blend", "-P", ROOT / "rigging/blender/retarget_kimodo.py", "--", mapping,
             d / "kimodo.npz", rig / "rig_info.json", d / "motion.json", *(["--speed", speed] if speed != 1 else [])],
            cwd=ROOT, stdout=subprocess.DEVNULL)
        how = "kimodo"
    else:
        run([BLENDER, "-b", rig / "rigged.blend", "-P", ROOT / "rigging/blender/retarget_rig.py", "--", mapping,
             srig / "rig_info.json", s / "motion.json", rig / "rig_info.json", d / "motion.json"],
            cwd=ROOT, stdout=subprocess.DEVNULL)
        how = "rig"
    springs = ["--springs", rig / "springs.json"] if (rig / "springs.json").exists() else []
    with open(d / "blender.log", "w") as log:
        run([BLENDER, "-b", "-P", "blender/apply_motion.py", "--", rig / "rigged.blend", rig / "rig_info.json",
             d / "motion.json", d / "render", "--glb", *springs], cwd=ROOT / "external/anymove", stdout=log,
            stderr=subprocess.STDOUT)
    promote = rig.parent / "promote.json"
    source.update(rig={
        "model": json.loads(promote.read_text()).get("model") if promote.exists() else None,
        "built": time.strftime("%Y-%m-%d %H:%M", time.localtime((rig / "rigged.blend").stat().st_mtime))},
        copied_from={"capture": str(s.relative_to(src.parent.parent)), "retargeted_from": how})
    (d / "source.json").write_text(json.dumps(source, indent=1))
    print(f"capture {d}", flush=True)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    if not (dst / "rigging/rigged/rigged.blend").exists():
        sys.exit(f"no rig in {dst / 'rigging/rigged'}: promote a model first (rigging/promote.py)")
    for name in sys.argv[3:]:
        copy(src, dst, name)


if __name__ == "__main__":
    main()
