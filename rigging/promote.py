"""Make a TRELLIS model the character's mesh: prep it into the project's anymove/ and rig it into rigging/.

    python -m rigging.promote PROJECT_DIR MODEL.glb [--height 1.75] [--no-rig]

One project is one character: its views/ and TRELLIS runs sit beside the rigged mesh. The project's current anymove/
and rigging/ (mesh, rig, captures) move to archive/<date-time>/ first, so nothing is overwritten; a capture belongs to
the rig it was made with, so it goes with it. Carried over to the new mesh: anymove/creature.json (with --height as its
size if given), the moves' prompts (anymove/prompts/), and rigging/character.json without anything tied to the old
mesh's coordinates (prop tips are found again on the new mesh). Without them, generic defaults.
Then anymove's prep (scale, stand on the ground, grid views), rigging/rig.sh and rigging/ref_frame.sh (the
limb-painted two-camera frame a Wan clip starts from, anymove/runs/ref/ref.png). Rigging uses the GPU and a lot of
memory: run it in a cap (the studio does), e.g. systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREP_PY = os.environ.get("ANYMOVE_PYTHON", str(Path.home() / "venvs/sam3d-xpu/bin/python"))  # needs PIL
LAYOUT = {"H": 720, "W_FRONT": 640, "W_SIDE": 640, "M_PER_PX": 0.0036, "GROUND_V": 640}


def default_creature(name, height):
    return {"name": name, "noun": name, "prep": {"length_m": height, "rotz_deg": 0, "axis": "z"}, "layout": dict(LAYOUT)}


def default_character(name):
    hold = lambda s: {"parent": f"hand_{s}", "tip_kp": f"bladetip_{s}"}  # an empty hand is found empty: no prop
    return {"name": name, "template": "humanoid", "static": "../anymove/prep/static.blend",
            "props": {"weapon_L": hold("L"), "weapon_R": hold("R")}, "hair": {"strand_parent": "spine_03"},
            "layout": dict(LAYOUT, H=720), "camera": {"cam": "-2.6,-3.0,1.4", "target": "0,0,0.95", "lens": 45}}


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("model")
    ap.add_argument("--height", type=float, help="the character's real height in metres (default: kept, else 1.75)")
    ap.add_argument("--no-rig", action="store_true", help="prep the mesh only")
    a = ap.parse_args()
    p, model = Path(a.project).resolve(), Path(a.model).resolve()
    if not model.is_file() or model.suffix != ".glb":
        sys.exit(f"not a .glb: {model}")
    am, rg = p / "anymove", p / "rigging"
    creature = json.loads((am / "creature.json").read_text()) if (am / "creature.json").exists() else None
    character = json.loads((rg / "character.json").read_text()) if (rg / "character.json").exists() else None

    archived = None
    if am.exists() or rg.exists():
        archived = p / "archive" / time.strftime("%Y%m%d-%H%M%S")
        archived.mkdir(parents=True)
        for d in (am, rg):
            if d.exists():
                shutil.move(str(d), str(archived / d.name))
        print(f"the previous mesh and rig: {archived}")

    am.mkdir()
    shutil.copy(model, am / "model.glb")
    creature = creature or default_creature(p.name, a.height or 1.75)
    if a.height:
        creature.setdefault("prep", {})["length_m"] = a.height
    (am / "creature.json").write_text(json.dumps(creature, indent=2))
    if archived and (archived / "anymove/prompts").is_dir():
        shutil.copytree(archived / "anymove/prompts", am / "prompts")
    run(["bash", ROOT / "external/anymove/tools/prep.sh", am], env={**os.environ, "PYTHON": PREP_PY})

    rg.mkdir()
    character = character or default_character(p.name)
    for spec in character.get("props", {}).values():
        spec.pop("tip", None)  # the old mesh's coordinates: found again on this one
    (rg / "character.json").write_text(json.dumps(character, indent=1))
    (rg / "promote.json").write_text(json.dumps({
        "model": os.path.relpath(model, p), "when": time.strftime("%Y-%m-%d %H:%M"),
        "archived": os.path.relpath(archived, p) if archived else None}, indent=1))
    if not a.no_rig:
        run(["bash", ROOT / "rigging/rig.sh", rg], cwd=ROOT)
        run(["bash", ROOT / "rigging/ref_frame.sh", p], cwd=ROOT)  # the frame a Wan clip starts from: anymove/runs/ref
    print(f"promoted {model} -> {am} and {rg}")


if __name__ == "__main__":
    main()
