"""A project's characters, clips and items for a browser game (three.js), with a manifest.

    python -m rigging.export_game PLAN.json

PLAN.json: {"project": PROJECT_DIR, "characters": {NAME: [CAPTURE, ...]}, "items": [NAME, ...], "copy_to": DIR}
Writes PROJECT_DIR/export/ (the previous export kept as export_previous/), and copies it to copy_to if given (a folder
in a game project, e.g. its client/assets/models/sparkdream): replaced only if it is empty or an earlier copy of this
project's export (its manifest.json names the project), so nothing else of the game's is ever overwritten.
    manifest.json
    characters/<name>/character.glb   meshes, skeleton and sockets, rest pose (rigging/blender/export_character.py)
    characters/<name>/clips/<capture>.glb   the skeleton's animation only, named <capture> (apply_motion.py --anim-only)
    items/<name>.glb                   the item's light copy, grip at its origin (rigging/item_game.py)
    equip.js, index.html               the studio's three.js module and an example page using it
A clip plays on its character by bone name; an item goes under a socket node with no transform of its own. The
manifest (units: metres, +y up, characters facing +z):
    {"characters": {NAME: {"model", "sockets": {SOCKET: {"fits": KIND}}, "clips": {CAPTURE: {"file", "move", "seconds",
      "loop", "in_place", "travel_per_cycle_m"}}, "loadout": {SOCKET: ITEM}}},
     "items": {NAME: {"model", "fits": [KIND], "length_m"}}}
"loadout" is the character's default from the studio's equip preview (rigging/loadout.json), for a game to use or
ignore. A socket's kind is its name's middle (socket_hand_R: hand); an item fits the kinds item.json lists.
"""
import json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLENDER = "blender"


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def copy_export(src, dest, project):
    """The export folder `src` copied to `dest` (see the module docstring), swapped in whole once copied."""
    dest = dest.expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        m = dest / "manifest.json"
        if not (m.exists() and json.loads(m.read_text()).get("project") == project):
            sys.exit(f"{dest} holds something other than an earlier copy of {project}'s export: not replaced "
                     f"(the export is in {src}; choose an empty or new folder)")
    tmp = dest.with_name(dest.name + ".new")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    print(f"export: copied to {dest}", flush=True)


def main():
    plan = json.loads(Path(sys.argv[1]).read_text())
    p = Path(plan["project"]).resolve()
    out = p / "export.new"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    manifest = {"format": 1, "project": p.name, "made": time.strftime("%Y-%m-%d %H:%M"), "units": "metres",
                "up": "+y", "facing": "+z", "characters": {}, "items": {}}
    items = {}
    for name in plan.get("items", []):
        d = p / "items" / name
        meta = json.loads((d / "item.json").read_text())
        if not meta.get("game"):
            sys.exit(f"{name} has no light copy yet: use one of its models for the item first")
        (out / "items").mkdir(exist_ok=True)
        shutil.copy(d / meta["game"]["model"], out / "items" / f"{name}.glb")
        items[name] = {"model": f"items/{name}.glb", "fits": meta.get("sockets", ["hand"]), "length_m": meta.get("length_m")}
        if not meta["game"].get("metres"):
            print(f"{name}: no real length given, so it is in the model's units (about one unit long)", flush=True)
    manifest["items"] = items

    for name, captures in plan.get("characters", {}).items():
        c = p / name
        rig = c / "rigging/rigged"
        info = json.loads((rig / "rig_info.json").read_text())
        co = out / "characters" / name
        (co / "clips").mkdir(parents=True)
        run([BLENDER, "-b", rig / "rigged.blend", "-P", ROOT / "rigging/blender/export_character.py", "--",
             co / "character.glb"], cwd=ROOT, stdout=subprocess.DEVNULL)
        springs = ["--springs", rig / "springs.json"] if (rig / "springs.json").exists() else []
        clips = {}
        for k in captures:
            kd = c / "anymove/runs/clips" / k
            tmp = out / ".tmp" / k
            run([BLENDER, "-b", "-P", "blender/apply_motion.py", "--", rig / "rigged.blend", rig / "rig_info.json",
                 kd / "motion.json", tmp, "--glb", "--anim-only", "--action", k, *springs],
                cwd=ROOT / "external/anymove", stdout=subprocess.DEVNULL)
            shutil.move(str(tmp / "animation.glb"), co / "clips" / f"{k}.glb")
            src = json.loads((kd / "source.json").read_text()) if (kd / "source.json").exists() else {}
            m = json.loads((kd / "motion.json").read_text())
            loop = src.get("loop") or {}
            clips[k] = {"file": f"characters/{name}/clips/{k}.glb", "move": src.get("move") or k,
                        "seconds": round(len(m["euler"]) / float(m.get("fps", 30)), 3), "loop": bool(loop),
                        "in_place": bool(loop.get("in_place")), "travel_per_cycle_m": loop.get("travel_per_cycle_m")}
        sockets = {s: {"fits": s.split("_")[1]} for s in info.get("sockets", {})}
        loadout = json.loads((c / "rigging/loadout.json").read_text()) if (c / "rigging/loadout.json").exists() else {}
        manifest["characters"][name] = {"model": f"characters/{name}/character.glb", "sockets": sockets, "clips": clips,
                                        "loadout": {s: i for s, i in loadout.items() if s in sockets and i in items}}
    shutil.rmtree(out / ".tmp", ignore_errors=True)
    shutil.copy(ROOT / "studio/equip.js", out / "equip.js")
    shutil.copy(ROOT / "studio/export_example.html", out / "index.html")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    old, prev = p / "export", p / "export_previous"
    if old.exists():
        shutil.rmtree(prev, ignore_errors=True)
        old.rename(prev)
    out.rename(old)
    size = sum(f.stat().st_size for f in old.rglob("*") if f.is_file())
    print(f"export: {len(manifest['characters'])} characters, "
          f"{sum(len(v['clips']) for v in manifest['characters'].values())} clips, {len(items)} items, "
          f"{size / 1e6:.1f} MB -> {old}", flush=True)
    if plan.get("copy_to"):
        copy_export(old, Path(plan["copy_to"]), p.name)


if __name__ == "__main__":
    main()
