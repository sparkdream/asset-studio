"""Captures of a move made from words with Kimodo, on a project's rig: text -> Kimodo samples -> each retargeted onto
the rig and exported as a capture the studio files under its move.

    python -m rigging.kimodo_capture PROJECT_DIR MOVE "A person ..." [--duration 2] [--samples 4] [--seed 0]
        [--pose TIME=CAPTURE_OR_NPZ[:FRAME][^HIPS_M][@DEG][+arms=...][+legs=...][+straight][+upright]] [--heading TIME=DEG]

    python -m rigging.kimodo_capture PROJECT_DIR MOVE --recipe [RECIPE.json] [--seed 0]

--pose pins the full body at TIME (seconds, or "end") to a frame of another Kimodo capture (its folder, or its
kimodo.npz), --heading where the hips face: see rigging/kimodo_gen.py. Both are recorded in source.json.

--recipe reads the prompt and the pins from a file (default: the move's anymove/prompts/<MOVE>.kimodo.json, which the
studio's Kimodo card edits). Capture names in it are folders of anymove/runs/clips, "NAME:FRAME" picks a frame:
    {"prompt": "A person ...", "duration": 3, "samples": 4,
     "poses": [{"at": 0.5, "from": "double_slash_kimodo_s61_1:24"},
               {"at": 2.0, "from": "land_lunge_kimodo_s81_2:25", "arms": "double_slash_kimodo_s61_1"}],
     "headings": [{"at": 1.1, "facing": 180}],
     "jump": {"apex": 1.1, "hips": 1.75, "every": 2, "crouch": {"from": "jump_kimodo_s31_0:20"},
              "spin": {"turns": 1, "seconds": 0.39},
              "pose": {"from": "jump_kimodo_s31_0:30", "arms": "...", "legs": "...", "straight": true, "upright": true}}}
A pose also takes "facing" (degrees), "hips" (their height, metres), "legs", "straight", "upright". "jump" pins its
"pose" every `every` frames of the flight, the hips on a ballistic arc peaking at `hips` metres at `apex` seconds
(the flight's length follows from the height, from standing hips of `stand` = 1.0 m), travelling "forward" metres
(along the start's facing, evenly over the flight; poses after the landing are pinned where it lands), turning
`turns` times in `seconds` centred on the apex (default: the whole flight) or from "start" seconds after the takeoff
(negative: the turn begins as the feet push off, which reads as springing into the spin), and "crouch" just before the takeoff.
"run_in" / "run_out" (m/s) keep the poses before the takeoff / after the landing moving at that speed (a running
jump), instead of on the takeoff / landing spot. "release":
{"arms": "CAPTURE:FRAME", "seconds": 0.2} sweeps the air pose's arms into those over the flight's last seconds,
speeding up to the touchdown: a strike that lands with the body; its "legs" (over "legs_seconds", default 0.25)
ease the air pose's legs, straightened or not, toward a landing's, so the knees bend before the feet touch;
"CAPTURE:FRAME*0.45" goes only that far by touchdown, leaving the rest of the bend to absorb the landing. "arms" (and "legs") may be a list applied in order,
and "CAPTURE:FRAME*0.5" goes only half way from the arms so far to that frame's.

Writes PROJECT_DIR/anymove/runs/clips/<MOVE>_kimodo_s<seed>_<i>/: kimodo.npz, motion.json, render/animation.glb,
source.json ({"move", "kimodo": prompt, ...}). The rig is the project's rigging/rigged/ (rigging/promote.py).
Short, mocap-caption-like prompts move best ("A person swings a sword horizontally from right to left with their right
hand."). Kimodo runs in $KIMODO_PYTHON (default ~/venvs/kimodo-xpu, see rigging/kimodo_gen.py) and takes about 17 GB of
GPU memory, and loads its 16 GB text encoder
through RAM first: run it in a memory cap of about 20 GB with no swap, as the studio does.
"""
import argparse, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KIMODO_PY = os.environ.get("KIMODO_PYTHON", str(Path.home() / "venvs/kimodo-xpu/bin/python"))
BLENDER = os.environ.get("BLENDER", "blender")


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


FPS = 30  # Kimodo's


def recipe_args(r, clips):
    """--pose / --heading strings from a recipe (see the docstring)."""
    def src(name):
        name, _, fr = str(name).partition(":")
        return str(clips / name) + (f":{fr}" if fr else "")

    def pose(at, q, facing=None, hips=None, ahead=None):
        s = f"{at}={src(q['from'])}"
        h = q.get("hips", hips)
        s += f"^{h:.3f}" if h is not None else ""
        f = q.get("facing", facing)
        s += f"@{f:.1f}" if f is not None else ""
        z = q.get("ahead", ahead)
        s += f">{z:.3f}" if z else ""
        for part in ("arms", "legs"):
            for x in ([q[part]] if isinstance(q.get(part), str) else q.get(part) or []):
                x, _, mix = str(x).partition("*")
                s += f"+{part}={src(x)}" + (f"*{mix}" if mix else "")
        for flag in ("straight", "upright"):
            if q.get(flag):
                s += f"+{flag}"
        for part, x in q.get("then", []):  # after the flags: a release's blend away from straightened knees
            x, _, mix = str(x).partition("*")
            s += f"+{part}={src(x)}" + (f"*{mix}" if mix else "")
        return s

    out = []
    j = r.get("jump")
    if j:
        g, apex, top, stand = 9.81, float(j["apex"]), float(j["hips"]), float(j.get("stand", 1.0))
        half = (2 * (top - stand) / g) ** 0.5
        takeoff, land = apex - half, apex + half
        fwd = float(j.get("forward", 0))  # metres the jump travels, evenly over the flight
        # "run_in"/"run_out" (m/s): poses before the takeoff / after the landing keep moving at that speed instead of
        # standing on the takeoff / landing spot (a running jump pinned to a run's strides otherwise stalls through
        # its push-off and stops dead after landing: the rogue's started at 1.3 m/s, which a game scaling its travel
        # by the starting speed blew up to 8 m)
        rin, rout = float(j.get("run_in", 0)), float(j.get("run_out", 0))
        ahead = lambda t: (fwd * min(max((t - takeoff) / (land - takeoff), 0), 1) - rin * max(takeoff - t, 0)
                           + rout * max(t - land, 0))
    else:
        fwd, ahead = 0.0, lambda t: 0.0
    end_t = float(r.get("duration", 2.0))
    for q in r.get("poses", []):
        z = ahead(end_t if q["at"] == "end" else float(q["at"]))  # poses after the landing are where it lands
        out += ["--pose", pose(q["at"], q, ahead=z)]
    for q in r.get("headings", []):
        z = ahead(end_t if q["at"] == "end" else float(q["at"]))
        out += ["--heading", f"{q['at']}={q['facing']}" + (f">{z:.3f}" if z else "")]
    if j:
        spin = j.get("spin") or {}
        turns = float(spin.get("turns", 0))
        secs = float(spin.get("seconds", land - takeoff))
        base = float(j.get("facing", 0))
        every = int(j.get("every", 2))
        f0, f1 = int(round(takeoff * FPS)) + 1, int(round(land * FPS)) - 1
        # the spin's start: "start" seconds from the takeoff (negative: already turning as the feet push off), else
        # centred on the apex
        s0 = takeoff + float(spin["start"]) if "start" in spin else apex - secs / 2
        for f in range(math.ceil(s0 * FPS - 1e-9), f0):  # turning on the ground, before the flight's pins: headings only
            t = f / FPS
            out += ["--heading", f"{t:.4f}={base + 360 * turns * min(max((t - s0) / secs, 0), 1):.1f}"]
        rel = j.get("release") or {}  # the air pose's arms let go into these over the flight's last seconds: a slash
        rsecs = float(rel.get("seconds", 0.2))
        lsecs = float(rel.get("legs_seconds", 0.25))
        frames = list(range(f0, f1 + 1, every))
        if rel and frames[-1] != f1:
            frames.append(f1)  # the release ends on the last frame in the air
        for f in frames:
            t = f / FPS
            k = min(max((t - s0) / secs, 0), 1)
            q = j["pose"]
            r = (t - (land - rsecs)) / rsecs if rel.get("arms") else -1
            if r > 0:
                r = min(r, 1) ** 2  # speeding up all the way through, like a strike
                arms = q.get("arms") or []
                q = {**q, "arms": ([arms] if isinstance(arms, str) else list(arms)) + [f"{rel['arms']}*{r:.3f}"]}
            r = (t - (land - lsecs)) / lsecs if rel.get("legs") else -1
            if r > 0:  # the legs reach for the landing: knees already bending at touchdown, to take it
                r = min(r, 1)
                legs, _, most = str(rel["legs"]).partition("*")  # "CAPTURE:FRAME*0.45": only that far by touchdown
                q = {**q, "then": [("legs", f"{legs}*{float(most or 1) * r * r * (3 - 2 * r):.3f}")]}
            out += ["--pose", pose(f"{t:.4f}", q, facing=base + 360 * turns * k, hips=top - g / 2 * (t - apex) ** 2,
                                   ahead=ahead(t))]
        if j.get("crouch"):
            out += ["--pose", pose(f"{max(takeoff - 0.15, 0):.4f}", j["crouch"], facing=base)]
        print(f"jump: flight {takeoff:.2f}-{land:.2f} s, pinned every {every} frames from frame {f0}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("move")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("--recipe", nargs="?", const="", default=None)
    ap.add_argument("--duration", type=float, default=2.0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose", action="append", default=[])
    ap.add_argument("--heading", action="append", default=[])
    a = ap.parse_args()
    p = Path(a.project).resolve()
    recipe = None
    if a.recipe is not None:
        rp = Path(a.recipe) if a.recipe else p / "anymove/prompts" / f"{a.move}.kimodo.json"
        recipe = json.loads(rp.read_text())
        extra = recipe_args(recipe, p / "anymove/runs/clips")
        a = ap.parse_args(sys.argv[1:] + extra)  # the recipe's pins after any on the command line
        a.prompt = recipe["prompt"]
        if "--duration" not in sys.argv:
            a.duration = float(recipe.get("duration", a.duration))
        if "--samples" not in sys.argv:
            a.samples = int(recipe.get("samples", a.samples))
    if not a.prompt:
        ap.error("a prompt, or --recipe")
    poses = []
    def npz(src):  # a capture folder stands for its kimodo.npz
        cut = min([i for i in (src.find(":"), src.find("^"), src.find("@"), src.find("*"), src.find(">")) if i >= 0] or [len(src)])
        path, sep, rest = src[:cut], src[cut:cut + 1], src[cut + 1:]
        path = Path(path).resolve()
        return f"{path / 'kimodo.npz' if path.is_dir() else path}{sep}{rest}"

    for spec in a.pose:
        spec, *extra = spec.split("+")
        t, src = spec.split("=", 1)
        poses.append(f"{t}={npz(src)}" + "".join("+" + (f"{x.split('=', 1)[0]}={npz(x.split('=', 1)[1])}" if "=" in x else x)
                                                 for x in extra))
    rig, clips = p / "rigging/rigged", p / "anymove/runs/clips"
    if not (rig / "rigged.blend").exists():
        sys.exit(f"no rig in {rig}: promote a model first (rigging/promote.py)")
    tmpl = json.loads((rig / "rig_info.json").read_text()).get("template", "humanoid")
    mapping = ROOT / "rigging/templates" / f"{tmpl}_kimodo.json"
    if not mapping.exists():
        sys.exit(f"no Kimodo map for the {tmpl} template ({mapping.name}): Kimodo only moves humanoids")

    gen = clips / f".kimodo_{a.move}_s{a.seed}"
    run([KIMODO_PY, "-m", "rigging.kimodo_gen", gen, a.prompt, "--duration", a.duration, "--samples", a.samples,
         "--seed", a.seed, *[x for v in poses for x in ("--pose", v)], *[x for v in a.heading for x in ("--heading", v)]],
        cwd=ROOT)
    springs = ["--springs", rig / "springs.json"] if (rig / "springs.json").exists() else []
    name = f"{a.move}_kimodo_s{a.seed}"  # a seed used again (other keyframes) never replaces its earlier captures
    for suffix in [""] + [chr(c) for c in range(ord("b"), ord("z") + 1)]:
        if not any(clips.glob(f"{name}{suffix}_*")):
            name += suffix
            break
    for i in range(a.samples):
        out = clips / f"{name}_{i}"
        out.mkdir(parents=True)
        shutil.move(str(gen / f"kimodo_{i:02d}.npz"), out / "kimodo.npz")
        run([BLENDER, "-b", rig / "rigged.blend", "-P", ROOT / "rigging/blender/retarget_kimodo.py", "--", mapping,
             out / "kimodo.npz", rig / "rig_info.json", out / "motion.json"], cwd=ROOT, stdout=subprocess.DEVNULL)
        with open(out / "blender.log", "w") as log:
            run([BLENDER, "-b", "-P", "blender/apply_motion.py", "--", rig / "rigged.blend", rig / "rig_info.json",
                 out / "motion.json", out / "render", "--glb", *springs], cwd=ROOT / "external/anymove", stdout=log,
                stderr=subprocess.STDOUT)
        built = time.strftime("%Y-%m-%d %H:%M", time.localtime((rig / "rigged.blend").stat().st_mtime))
        made_on = json.loads((rig.parent / "promote.json").read_text()).get("model") if (rig.parent / "promote.json").exists() else None
        (out / "source.json").write_text(json.dumps({"move": a.move, "kimodo": a.prompt, "seed": a.seed, "sample": i,
                                                     "duration": a.duration, "speed": 1,
                                                     **({"constraints": {"pose": poses, "heading": a.heading}}
                                                        if poses or a.heading else {}),
                                                     **({"recipe": recipe} if recipe else {}),
                                                     "rig": {"model": made_on, "built": built}}, indent=1))
        print(f"capture {out}", flush=True)
    shutil.rmtree(gen, ignore_errors=True)


if __name__ == "__main__":
    main()
