"""Local review studio: look at the TRELLIS.2 models, the Wan clips and the captures in outputs/ before going further,
let the checks find what went wrong where, and queue the runs that fix it.

    python3 studio/server.py [port=8765]        # then open http://localhost:8765 (WSL forwards it to Windows)

Standard library only. Jobs run in $STUDIO_PYTHON (default ~/venvs/trellis2-xpu) with env.sh sourced, each writing its
log next to its output, on two queues: one for the GPU (Wan, TRELLIS.2, reading clips, captures, renders) and one for
the checks that only need the CPU. Opening a project, and every finished job, queues the checks anything new needs
(studio/checks): the rig's skin weights, every capture (reading, fit, skin) and every two-camera Wan clip not yet
captured (reading). Verdicts and notes go to outputs/<project>/review.json.
Reference views for multi-view TRELLIS.2 (reference/views.py: DIR/views/) are made, edited and approved from the Views
tab; Qwen-Image-Edit runs in $QWEN_PYTHON (default ~/venvs/kimodo-xpu).
STUDIO_DRY_RUN=1 lists the jobs asked for, with their scripts, without running or saving any (for trying the page).
Browsers can't play the MPEG-4 Part 2 clips Wan writes, so each gets an H.264 copy beside it (<name>_h264.mp4, as
before), made in the background when a project is opened.
"""
import json, mimetypes, os, pathlib, queue, random, re, shlex, shutil, signal, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
STATE = OUT / ".studio"  # jobs.json and uploads
CHECKS = ROOT / "studio/checks"
PY = os.environ.get("STUDIO_PYTHON", str(pathlib.Path.home() / "venvs/trellis2-xpu/bin/python"))
BLENDER = os.environ.get("BLENDER", "blender")
# a Python with SAM 3D Body (mocap.body3d): depth for captures from the clip's front view alone
BODY3D_PY = os.environ.get("BODY3D_PYTHON", str(pathlib.Path.home() / "venvs/sam3d-xpu/bin/python"))
PAGE = pathlib.Path(__file__).with_name("index.html")
SKIP = {"dataset", "__pycache__", "archive", "export", "export_previous", "export.new"}  # 2,200 training poses; rigs a promotion replaced (rigging/promote.py); a project's game export (rigging/export_game.py)
PLAYABLE = {"h264", "vp9", "av1"}
WAN_MODELS = ["wan22-14b", "wan22"]                     # the first is the default
WAN_SECONDS = {"wan22-14b": 3, "wan22": 5}  # the longest clip that fits the 32 GB GPU: 5 s of the 14B spills, hours per clip
TRELLIS_TYPES = ["512", "1024", "1024_cascade", "1536_cascade"]
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# Jobs get this environment without what pyenv's python3 shim adds (its folders first on PATH, PYENV_*): with them,
# anymove's joint reader segfaults on the Intel GPU when started from this server.
JOB_ENV = {k: v for k, v in os.environ.items() if not k.startswith("PYENV_")}
JOB_ENV["PATH"] = os.pathsep.join(p for p in JOB_ENV.get("PATH", "").split(os.pathsep) if "/.pyenv/" not in p or p.endswith("/shims"))
STARTED = time.time()
QWEN_PY = os.environ.get("QWEN_PYTHON", str(pathlib.Path.home() / "venvs/kimodo-xpu/bin/python"))
DRY = bool(os.environ.get("STUDIO_DRY_RUN"))
sys.path.insert(0, str(ROOT))
from reference.views import ANGLES, SETS, Views  # noqa: E402  (standard library only at import time)
VIEWS = SETS["eight"] + SETS["turntable"][1:]  # every view a folder can hold, in the order the page shows them

mimetypes.add_type("model/gltf-binary", ".glb")


# A character is a folder in outputs/: on its own (outputs/knight), or in a project (outputs/sparkdream/rogue), a folder
# marked by its project.json that groups characters. Its id is that path ("knight", "sparkdream/rogue"); the API still
# calls it "project", from when every character was a top-level folder.
# An item (a dagger, a quiver: anything a character holds or wears) is shared by every character in its project, in
# the project's items/ folder (outputs/sparkdream/items/dagger, id "sparkdream/items/dagger"). It is made like a
# character, with its own views and TRELLIS runs, so the page treats it as one.
CHARACTER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")
ITEMS = "items"


def is_group(d):
    return (d / "project.json").is_file()


def character_id(name):
    """name, if it is a character's id (not a project's, nor a folder inside a character)."""
    name = str(name).strip().strip("/")
    parts = name.split("/")
    if len(parts) == 3 and parts[1] == ITEMS and is_group(OUT / parts[0]) and CHARACTER.match(parts[2]) and not SKIP & set(parts):
        return name
    if len(parts) == 2 and parts[1] == ITEMS and is_group(OUT / parts[0]):
        raise ValueError(f"not a character: {name} (a project's items folder)")
    if not CHARACTER.match(name) or SKIP & set(parts) or is_group(OUT / name) or (len(parts) == 2) != is_group(OUT / parts[0]):
        raise ValueError(f"not a character: {name}")
    return name


def owner(p):
    """The id of the character that path p (under outputs/) belongs to."""
    parts = pathlib.Path(p).resolve().relative_to(OUT.resolve()).parts
    if len(parts) > 2 and parts[1] == ITEMS and is_group(OUT / parts[0]):
        return "/".join(parts[:3])
    return "/".join(parts[:2] if len(parts) > 1 and is_group(OUT / parts[0]) else parts[:1])


def listing():
    """Every character (items included: project/items/name), and every project (possibly still empty)."""
    chars, groups = [], []
    sub = lambda d: [c for c in sorted(d.iterdir()) if c.is_dir() and not c.name.startswith(".") and c.name not in SKIP]
    for d in sorted(OUT.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name in SKIP:
            continue
        if is_group(d):
            groups.append(d.name)
            for c in sub(d):
                chars += [f"{d.name}/{ITEMS}/{i.name}" for i in sub(c)] if c.name == ITEMS else [f"{d.name}/{c.name}"]
        else:
            chars.append(d.name)
    return chars, groups


def inside(base, rel):
    """base/rel, refusing anything that resolves outside base."""
    p = (base / rel).resolve()
    if p != base.resolve() and base.resolve() not in p.parents:
        raise ValueError(f"outside {base}: {rel}")
    return p


def rel(p):
    return p.relative_to(OUT).as_posix()


def info(p):
    st = p.stat()
    return {"path": rel(p), "name": p.name, "size": st.st_size, "mtime": st.st_mtime}


def read_json(p, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def newer(out, inputs):
    """out exists and is newer than every input."""
    return out.exists() and out.stat().st_mtime >= max((p.stat().st_mtime for p in inputs if p.exists()), default=0)


# ---------------------------------------------------------------- videos the browser can play

_probes, _locks, _lock = {}, {}, threading.Lock()
transcodes = queue.Queue()


def probe(p):
    """codec, width and height of a video's first stream."""
    key = (p, p.stat().st_mtime)
    if key not in _probes:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=codec_name,width,height", "-of", "csv=p=0", str(p)], capture_output=True, text=True)
        parts = (r.stdout.strip().split(",") + ["", "0", "0"])[:3]
        _probes[key] = {"codec": parts[0], "width": int(parts[1] or 0), "height": int(parts[2] or 0)}
    return _probes[key]


def two_camera(p):
    """A front-and-side clip: wider than 1.5 : 1 (single views are 640 x 720)."""
    v = probe(p)
    return v["height"] > 0 and v["width"] / v["height"] > 1.5


def h264_copy(p):
    return p.with_name(p.stem + "_h264.mp4")


def playable(p):
    """p, or its H.264 copy, made now if missing or older than p."""
    if probe(p)["codec"] in PLAYABLE:
        return p
    copy = h264_copy(p)
    with _lock:
        lock = _locks.setdefault(copy, threading.Lock())
    with lock:  # the browser asks for a clip several times at once
        if not newer(copy, [p]):
            tmp = copy.with_suffix(".tmp.mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(p), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "18", "-movflags", "+faststart", "-an", str(tmp)], check=True)
            tmp.replace(copy)
    return copy


def transcoder():
    while True:
        p = transcodes.get()
        try:
            playable(p)
        except Exception as e:
            print("transcode failed:", p, e, file=sys.stderr)


# ---------------------------------------------------------------- what a project holds

def creatures(project):
    """anymove creature folders (with a creature.json) inside a project."""
    return sorted(p.parent for p in project.rglob("creature.json") if not SKIP & set(p.parts))


def log_for(p):
    for name in (p.stem + ".log", "wan.log", "run.log"):
        if (p.parent / name).exists():
            return rel(p.parent / name)


def clip_check_dir(v):
    return v.parent / (v.stem + ".check")


RIG_CODE = [ROOT / "rigging/rig.sh", ROOT / "rigging/fit.py", ROOT / "rigging/segment.py", ROOT / "external/anymove/blender/add_hair.py",
            *(ROOT / "rigging/blender").glob("*.py"), *(ROOT / "rigging/templates").glob("*.json")]


def rig_code_newer(rigged):
    """Whether the rigging code changed after this rig was built: only then can a rebuild give different weights."""
    return rigged.exists() and any(p.exists() and p.stat().st_mtime > rigged.stat().st_mtime for p in RIG_CODE)


def rig_of(c):
    """Where a creature's rig is: (rigged dir, character dir). In the one-project layout (rigging/promote.py) that is
    rigging/rigged/ beside the creature's anymove/, built by rigging/rig.sh; older projects keep anymove's own rigged/
    (character dir None)."""
    ch = c.parent / "rigging"
    return (ch / "rigged", ch) if (ch / "character.json").exists() else (c / "rigged", None)


def rig_record(c):
    """What a capture is made on, for its source.json: the rig's source model and when the rig was built."""
    rd, ch = rig_of(c)
    b = rd / "rigged.blend"
    return {"model": read_json(ch / "promote.json", {}).get("model") if ch else None,
            "built": time.strftime("%Y-%m-%d %H:%M", time.localtime(b.stat().st_mtime)) if b.exists() else None}


def capture_info(k):
    """A capture folder (runs/clips/NAME): its videos, animation, check and the Wan clip it was made from.
    One not made from a clip (a Kimodo motion) names its move and prompt in source.json: "move", "kimodo"."""
    f = lambda name: rel(k / name) if (k / name).exists() else None
    check = read_json(k / "check.json")
    source = read_json(k / "source.json") or {}
    src = source.get("video") or (check or {}).get("source")
    if src:
        src = pathlib.Path(src)
        src = rel(src if src.is_absolute() else ROOT / src)
    mtimes = [p.stat().st_mtime for p in k.rglob("*") if p.is_file() and p.suffix in (".glb", ".mp4", ".json")]
    try:  # exported before the rig was last built: its skin is out of date (export it again)
        rig = rig_of(creature_of(k))[0] / "rigged.blend"
        rig_stale = (k / "render/animation.glb").exists() and rig.exists() and newer(rig, [k / "render/animation.glb"])
    except ValueError:
        rig_stale = False
    rig = source.get("rig")
    if not rig:  # made before captures recorded it: the current rig's model, or (a project from before promotion)
        try:    # the creature's own mesh, which every rig of it was built on
            c = creature_of(k)
            ch = rig_of(c)[1]
            src = read_json(ch / "promote.json", {}).get("model") if ch else None
            if not src and (c / "model.glb").exists():
                src = (c / "model.glb").relative_to(c.parent).as_posix()
            rig = {"model": src, "assumed": True} if src else None
        except ValueError:
            rig = None
    return {"name": k.name, "rig_stale": bool(rig_stale), "rig": rig, "loop": source.get("loop"), "dir": rel(k), "showcase": f("showcase.mp4"), "overlay": f("overlay.mp4"),
            "glb": f("render/animation.glb"), "check": check, "source": src, "mtime": max(mtimes, default=0),
            "speed": source.get("speed", 1), "body3d": bool(source.get("body3d")),
            "move": source.get("move"), "kimodo": source.get("kimodo"),
            "checking": not newer(k / "check.json", [k / "render/animation.glb"])}


def scan(name):
    project = inside(OUT, character_id(name))
    crs = creatures(project)
    models, others, moves, clips, caps = [], [], {}, {}, []

    def move(c, m):
        key = f"{rel(c)}:{m}"
        if key not in moves:
            prompt = c / "prompts" / f"{m}.txt"
            moves[key] = {"key": key, "creature": rel(c), "move": m, "clips": [], "captures": [],
                          "prompt": prompt.read_text().strip() if prompt.exists() else ""}
        return moves[key]

    for d, dirs, files in os.walk(project):
        dirs[:] = sorted(x for x in dirs if x not in SKIP and not x.startswith(".") and not x.endswith(".check"))
        d = pathlib.Path(d)
        c = next((c for c in crs if c == d or c in d.parents), None)
        where = d.relative_to(c / "runs").parts if c and (c / "runs") in [d, *d.parents] else ()
        if where[:1] == ("clips",):
            if len(where) == 2 and (d / "render/animation.glb").exists():
                caps.append({**capture_info(d), "creature": rel(c)})  # placed under a move once the clips are known
            continue
        for f in sorted(files):
            p = d / f
            if p.suffix not in (".glb", ".mp4") or f.endswith(".tmp.mp4"):
                continue
            if f.endswith("_h264.mp4") and p.with_name(f[:-9] + ".mp4").exists():  # shown as its original
                continue
            item = info(p)
            if p.suffix == ".mp4":
                item["log"] = log_for(p)
                if probe(p)["codec"] not in PLAYABLE and not h264_copy(p).exists():
                    transcodes.put(p)
            if where[:1] == ("videos",) and len(where) >= 2 and p.suffix == ".mp4":
                item["label"] = "/".join(where[2:] + (p.stem,))
                snap = p.with_suffix(".prompt.txt")
                item["prompt"] = snap.read_text().strip() if snap.exists() else None
                item["job"] = read_json(p.with_suffix(".job.json"))
                item["two_camera"] = two_camera(p)
                cd = clip_check_dir(p)
                item["check"] = read_json(cd / "check.json")
                item["overlay"] = rel(cd / "overlay.mp4") if (cd / "overlay.mp4").exists() else None
                item["captures"] = []
                clips[item["path"]] = item
                move(c, where[1])["clips"].append(item)
            elif p.suffix == ".glb":
                item["group"] = p.parent.relative_to(project).as_posix()
                item["made_from"] = read_json(p.parent / "views.json")  # the set and view versions, or the image
                item["raw"] = f == "sample_xpu.glb"  # TRELLIS's untextured mesh, a by-product of the textured one
                item["optimised"] = read_json(p.parent / "optimise.json")  # an optimised copy (rigging/optimise.py)
                item["layout"] = rel(p.parent / "layout.png") if (p.parent / "layout.png").exists() else None
                item["optimisable"] = (p.parent / "sample_xpu.pt").exists() and not item["raw"]
                item["paintable"] = (not item["raw"] and not p.stem.endswith("_reprojected") and
                                     p.relative_to(project).parts[0] not in ("anymove", "rigging", "archive", "game") and
                                     not p.with_name(f"{p.stem}_reprojected.glb").exists() and paint_run(p.parent) is not None)
                models.append(item)
            else:
                item["group"] = p.parent.relative_to(project).as_posix()
                others.append(item)
    for cap in caps:
        clip = clips.get(cap["source"])
        if clip:  # the clip shows what its capture's check found in the video
            clip["captures"].append(cap["name"])
            if cap["check"] and not clip["check"]:
                ch = cap["check"]
                clip["check"] = {**ch, "issues": [i for i in ch["issues"] if i["stage"] in ("video", "reading")],
                                 "from_capture": cap["name"], "score": ch.get("video_score"), "verdict": ch.get("video_verdict")}
            if not clip["overlay"]:
                clip["overlay"] = cap["overlay"]
            m = move(OUT / cap["creature"], pathlib.PurePosixPath(cap["source"]).relative_to(cap["creature"] + "/runs/videos").parts[0])
        elif cap["move"]:  # not made from a clip: it names its move
            m = move(OUT / cap["creature"], cap["move"])
        elif cap["source"] and cap["source"].startswith(cap["creature"] + "/runs/videos/"):  # its clip is gone
            m = move(OUT / cap["creature"], pathlib.PurePosixPath(cap["source"]).relative_to(cap["creature"] + "/runs/videos").parts[0])
        else:
            m = move(OUT / cap["creature"], cap["name"])
        m["captures"].append(cap)
    for c in crs:  # moves with a prompt and no clips yet
        for t in sorted((c / "prompts").glob("*.txt")):
            move(c, t.stem)
    for m in moves.values():
        m["mtime"] = max([x["mtime"] for x in m["clips"] + m["captures"]], default=0)
    images = [info(p) for d in (project, project / "inputs") if d.is_dir()
              for p in sorted(d.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    refs = [info(p) for c in crs for p in sorted((c / "runs").glob("ref*/ref.png"))]
    rigs = [{"creature": rel(c), "check": read_json(rig_of(c)[0] / "rig_check.json"),
             "image": rel(rig_of(c)[0] / "rig_check.png") if (rig_of(c)[0] / "rig_check.png").exists() else None,
             "todo": (c / "TODO.md").read_text() if (c / "TODO.md").exists() else "",
             "built": (rig_of(c)[0] / "rigged.blend").stat().st_mtime if (rig_of(c)[0] / "rigged.blend").exists() else None,
             "layout": "rigging" if rig_of(c)[1] else "anymove",
             "code_newer": rig_code_newer(rig_of(c)[0] / "rigged.blend") if rig_of(c)[1] else None}
            for c in crs]
    for c in crs:  # the rig's source model, and the rig's own copy of it (anymove/model.glb)
        ch = rig_of(c)[1]
        rec = read_json(ch / "promote.json", {}) if ch else {}
        src, tex = rec.get("model"), rec.get("texture") or rec.get("model")
        for m in models:
            if src and m["path"] == rel(project / src):
                m["rig_source"] = rel(c)
            if src and pathlib.Path(m["path"]).parent == pathlib.Path(rel(project / src)).parent and not m["raw"]:
                m["rig_paint"] = "wears" if m["path"] == rel(project / tex) else "can"  # same run: same mesh, other paint
            if m["path"] == rel(c / "model.glb"):
                m["rig_copy"] = rel(project / src) if src else True
    autoqueue(name)
    return {"name": name, "models": models, "videos": others, "images": images, "refs": refs, "rigs": rigs,
            "creatures": [rel(c) for c in crs], "moves": sorted(moves.values(), key=lambda m: -m["mtime"]),
            "review": read_json(project / "review.json", {}), "item": read_json(project / "item.json")}


def set_review(body):
    project = inside(OUT, body["project"])
    path = project / "review.json"
    with _lock:
        data = read_json(path, {})
        key = body["path"]
        inside(OUT, key)
        entry = {k: body.get(k) for k in ("verdict", "note") if body.get(k)}
        if entry:
            data[key] = {**entry, "updated": time.time()}
        else:
            data.pop(key, None)
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
    return data


def set_reviews(body):
    """One verdict (or none: cleared) for several paths at once; each keeps its note."""
    project = inside(OUT, body["project"])
    path = project / "review.json"
    if body.get("verdict") not in ("keep", "reject", None):
        raise ValueError("verdict: keep, reject or none")
    with _lock:
        data = read_json(path, {})
        for key in body["paths"]:
            inside(OUT, key)
            entry = {k: v for k, v in {"verdict": body.get("verdict"), "note": data.get(key, {}).get("note")}.items() if v}
            if entry:
                data[key] = {**entry, "updated": time.time()}
            else:
                data.pop(key, None)
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
    return data


# ---------------------------------------------------------------- deleting: into the project's .trash, never erased


def clip_files(v):
    """A Wan clip and what belongs to it (its H.264 copy, check folder, prompt and job snapshots, log); its whole
    folder when nothing else is in it."""
    mine = [x for x in v.parent.iterdir() if x.name.startswith(v.stem + ".") or x.name == v.stem + "_h264.mp4"]
    rest = [x for x in v.parent.iterdir() if x not in mine and x.name not in ("wan.log", "run.log")]
    return mine if rest else [v.parent]


def delete(body):
    """Move clips, captures or whole moves to <project>/.trash/<time>/, at the same place under the project, so
    anything can be put back by moving it back. items: {"kind": "clip", "path"} | {"kind": "capture", "path": dir} |
    {"kind": "move", "creature", "move"} (its clips folder and prompt; its captures are listed as captures)."""
    project = inside(OUT, body["project"])
    paths = []
    for it in body["items"]:
        if it["kind"] == "clip":
            v = inside(OUT, it["path"])
            if v.suffix != ".mp4" or "videos" not in v.parts:
                raise ValueError(f"not a clip: {it['path']}")
            paths += clip_files(v)
        elif it["kind"] == "capture":
            k = inside(OUT, it["path"])
            if k.parent.name != "clips":
                raise ValueError(f"not a capture: {it['path']}")
            paths.append(k)
        elif it["kind"] == "move":
            c = inside(OUT, it["creature"])
            m = pathlib.PurePath(it["move"]).name
            paths += [c / "runs/videos" / m, c / "prompts" / f"{m}.txt"]
    paths = [p for p in dict.fromkeys(paths) if p.exists()]
    if any(project not in p.parents for p in paths):
        raise ValueError("can only delete inside the open project")
    busy = [j for j in jobs if j.get("status") in ("queued", "running")
            and any(j["out"] == rel(p) or j["out"].startswith(rel(p) + "/") for p in paths)]
    if busy and not body.get("cancel_jobs"):  # the page asks, then sends cancel_jobs: true
        return {"busy": [{"id": j["id"], "label": j["label"], "status": j["status"], "auto": bool(j.get("auto"))} for j in busy]}
    if busy:
        procs = [j.get("proc") for j in busy if j["status"] == "running" and j.get("proc")]
        cancel_many({"ids": [j["id"] for j in busy]})
        for proc in procs:  # let them go before their folders move, or a dying one writes its log back into place
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                raise ValueError("a cancelled job is still stopping: try the delete again in a moment")
    stamp = project / ".trash" / time.strftime("%Y%m%d_%H%M%S")
    n = 1
    while stamp.exists():  # one folder per delete, even two in the same second
        stamp, n = stamp.with_name(time.strftime("%Y%m%d_%H%M%S") + f"_{n}"), n + 1
    for p in paths:
        dest = stamp / p.relative_to(project)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(dest))
    gone = [rel(p) for p in paths]
    with _lock:  # their keep/reject marks go with them
        path = project / "review.json"
        data = read_json(path, {})
        for key in [k for k in data if any(k == g or k.startswith(g + "/") for g in gone)]:
            data.pop(key)
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
    return {"moved": gone, "trash": rel(stamp), "cancelled": len(busy)}


# ---------------------------------------------------------------- jobs: one queue for the GPU, one for CPU checks

jobs, queues = [], {"gpu": queue.Queue(), "cpu": queue.Queue()}
_jobs_file = STATE / "jobs.json"
_jobs_lock = threading.Lock()


def save_jobs():
    if DRY:  # a trial server leaves the real job list alone
        return
    STATE.mkdir(parents=True, exist_ok=True)
    with _jobs_lock:
        _jobs_file.write_text(json.dumps([{k: v for k, v in j.items() if k != "proc"} for j in jobs[-300:]], indent=1))


def started_at(pid):
    """A process's start time in clock ticks since boot: with its pid, it names that one process."""
    try:
        return int((pathlib.Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def stop_group(job):
    """End a job's process group, if its leader is still the process the job started."""
    pgid = job.get("pgid")
    if pgid and job.get("pstart") and started_at(pgid) == job["pstart"]:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except OSError:
            pass


def load_jobs():
    for j in read_json(_jobs_file, []):
        if j["status"] in ("queued", "running") and not DRY:
            j["status"] = "interrupted"  # the server stopped before it finished
            stop_group(j)  # a job outlives a killed server: it runs in its own session
        jobs.append(j)


def submit(kind, label, project, script, log, out, lane="gpu", key=None, auto=False):
    job = {"id": f"{int(time.time() * 1000):x}{random.randrange(100):02d}", "kind": kind, "label": label,
           "project": project, "script": script, "log": rel(log), "out": rel(out), "status": "queued",
           "created": time.time(), "lane": lane, "key": key, "auto": auto}
    jobs.append(job)
    if DRY:
        job["status"] = "dry run"
        return job
    save_jobs()
    queues[lane].put(job)
    return job


def runner(lane):
    while True:
        job = queues[lane].get()
        if job["status"] != "queued":
            continue
        log = OUT / job["log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as f:
            f.write(f"$ {job['script']}\n".encode())
            f.flush()
            proc = subprocess.Popen(["bash", "-c", job["script"]], cwd=ROOT, env=JOB_ENV, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        job.update(status="running", started=time.time(), proc=proc, pgid=proc.pid, pstart=started_at(proc.pid))
        save_jobs()
        code = proc.wait()
        if job["status"] == "running":
            job["status"] = "done" if code == 0 else "failed"
        job.update(ended=time.time(), returncode=code)
        job.pop("proc", None)
        if job["status"] == "done" and job["kind"] in ("wan", "turntable"):  # its H.264 copy, ready when the page next looks
            transcodes.put(OUT / job["out"])
        save_jobs()
        try:
            autoqueue(job["project"])  # whatever the job changed gets checked
        except Exception as e:
            print("autoqueue:", e, file=sys.stderr)


def cancel(job_id):
    for j in jobs:
        if j["id"] == job_id and j["status"] in ("queued", "running"):
            if j["status"] == "running" and j.get("proc"):
                os.killpg(j["proc"].pid, signal.SIGTERM)
            j["status"] = "cancelled"
            save_jobs()
            return j
    raise ValueError(f"no queued or running job {job_id}")


def move_checks(body):
    """The automatic capture checks still waiting for a move's captures (source.json says which move): the ones a new
    generation of that move would make moot. A cancelled check isn't queued again until its capture changes."""
    out = []
    for j in jobs:
        if j["status"] != "queued" or not j.get("auto") or j.get("kind") != "check" or j.get("project") != body["project"]:
            continue
        src = read_json((OUT / j["out"]).parent / "source.json", {})
        if src.get("move") == body["move"]:
            out.append({"id": j["id"], "label": j["label"]})
    return out


def cancel_many(body):
    """Cancel the jobs in body["ids"], or with body["all"] every queued or running one (only the automatic ones with
    body["auto"]). Queued ones go first, so the queue doesn't start one of them as a running one is stopped; jobs that
    have meanwhile finished are skipped."""
    ids = None if body.get("all") else set(body.get("ids") or [])
    if ids is not None and not ids:
        raise ValueError("no jobs chosen")
    todo = [j for j in jobs if j["status"] in ("queued", "running") and (ids is None or j["id"] in ids)
            and (not body.get("auto") or j.get("auto"))]
    done = []
    for j in todo:
        if j["status"] == "queued":
            j["status"] = "cancelled"
            done.append(j)
    for j in todo:
        if j["status"] == "running":
            try:
                if j.get("proc"):
                    os.killpg(j["proc"].pid, signal.SIGTERM)
            except OSError:
                pass  # it ended just now
            j["status"] = "cancelled"
            done.append(j)
    save_jobs()
    return {"cancelled": len(done), "ids": [j["id"] for j in done]}


def tail(path, n=6000):
    """The end of a log, with tqdm's carriage-return redraws collapsed, and the last step count it shows."""
    p = OUT / path
    if not p.exists():
        return "", None
    with open(p, "rb") as f:
        f.seek(max(0, p.stat().st_size - n))
        text = f.read().decode(errors="replace")
    lines = [line.split("\r")[-1] if line.split("\r")[-1].strip() else line for line in text.split("\n")]
    steps = re.findall(r"(\d+)/(\d+) \[", text)
    return "\n".join(lines[-40:]), [int(x) for x in steps[-1]] if steps else None


def job_view(j):
    out = {k: v for k, v in j.items() if k != "proc" and (k != "script" or DRY)}
    out["tail"], out["steps"] = tail(j["log"])
    return out


def q(*args):
    return " ".join(shlex.quote(str(a)) for a in args)


def anymove(c):
    """The start of a script that runs anymove's tools for creature folder c."""
    return f"source env.sh && export MOCAP_CREATURE={q(c)} PYTHONPATH={q(ROOT / 'external/anymove')} && cd external/anymove && "


# ---------------------------------------------------------------- checks, queued by themselves

_autoqueue_lock = threading.Lock()


def auto(kind, key, inputs, *args, **kw):
    """Queue a check unless the same one is waiting or running, or failed since its inputs last changed (a restart of
    the server tries those again)."""
    changed = max([p.stat().st_mtime for p in inputs if p.exists()] + [STARTED])
    for j in jobs:
        if j.get("key") == key and (j["status"] in ("queued", "running", "dry run") or
                                    j["status"] in ("failed", "cancelled") and j["created"] > changed):
            return None
    return submit(kind, *args, key=key, auto=True, **kw)


def autoqueue(name):
    project = OUT / name
    if not project.is_dir():
        return
    with _autoqueue_lock:
        for c in creatures(project):
            rd = rig_of(c)[0]
            rigged = rd / "rigged.blend"
            if not rigged.exists():
                continue
            rc = rd / "rig_check.json"
            if not newer(rc, [rigged]):
                auto("rigcheck", f"rigcheck:{rel(c)}", [rigged], f"rig check · {c.name}", name,
                     f"source env.sh && {q(BLENDER)} -b -P {q(CHECKS / 'rig_check.py')} -- "
                     f"{q(rigged, c / 'creature.json', rc, rd / 'rig_check.png')}", rd / "rig_check.log", rc,
                     lane="cpu")
                continue  # the captures' verdicts depend on it
            rig_ok = "yes" if read_json(rc, {}).get("ok") else "no"
            waiting = False
            for glb in sorted((c / "runs/clips").glob("*/render/animation.glb")):
                k = glb.parent.parent
                inputs = [glb, k / "motion.json", k / "det.npz", rc, CHECKS / "check.py", CHECKS / "skin_check.py"]
                if newer(k / "check.json", inputs):
                    continue
                waiting = True
                det = f"--det {q(k / 'det.npz')} " if (k / "det.npz").exists() else ""
                find = (f"--find-source {q(c / 'runs/videos')} --overlay {q(k / 'overlay.mp4')} "
                        if det and (k / "overlay.mp4").exists() and not (k / "source.json").exists() else "")
                auto("check", f"check:{rel(k)}", inputs, f"check capture · {k.name}", name,
                     anymove(c) + f"cd {q(ROOT)} && ([ {q(k / 'skin.json')} -nt {q(glb)} ] || {q(BLENDER)} -b -P "
                     f"{q(CHECKS / 'skin_check.py')} -- {q(glb, c / 'creature.json', k / 'skin.json')}) && "
                     f"{q(PY)} {q(CHECKS / 'check.py')} {q(c, k / 'check.json')} {det}--motion {q(k / 'motion.json')} "
                     f"--skin {q(k / 'skin.json')} --rig-ok {rig_ok} {find}", k / "check.log", k / "check.json", lane="cpu")
            reader = c / "runs/reader.pt"
            if waiting or not reader.exists():
                continue  # clips get read once the captures say which clips they already cover
            covered = set()
            for k in (c / "runs/clips").glob("*/"):
                cap = capture_info(k) if (k / "render/animation.glb").exists() else None
                if cap and cap["source"] and cap["check"]:
                    covered.add(cap["source"])
            for v in sorted((c / "runs/videos").rglob("*.mp4")):
                if v.name.endswith(("_h264.mp4", ".tmp.mp4")) or v.parent.name.endswith(".check") or rel(v) in covered:
                    continue
                if not two_camera(v):
                    continue
                d = clip_check_dir(v)
                inputs = [v, reader, CHECKS / "check.py"]
                if newer(d / "check.json", inputs):
                    continue
                ref = (read_json(v.with_suffix(".job.json")) or {}).get("ref")
                ref = OUT / ref if ref else c / "runs/ref/ref.png"
                auto("clipcheck", f"clipcheck:{rel(v)}", inputs, f"read clip · {rel(v.parent)}/{v.stem}", name,
                     anymove(c) + f"mkdir -p {q(d)} && {q(PY)} -m mocap.detect {q(v, reader, ref, d / 'det.npz', d / 'overlay.mp4')} && "
                     f"cd {q(ROOT)} && {q(PY)} {q(CHECKS / 'check.py')} {q(c, d / 'check.json')} --det {q(d / 'det.npz')}",
                     d / "check.log", d / "check.json", lane="gpu")


# ---------------------------------------------------------------- jobs asked for from the page

def framed(creature, action, seconds, ref):
    """A move's action inside anymove's two-camera prompt (tools/make_prompt.py) when the reference is the split-screen
    frame and the prompt doesn't already describe it: told only the action, the video model drifts the side panel into
    a second front view (or swaps the panels), and the capture loses its depth."""
    with open(ref, "rb") as f:  # a PNG's size, from its header
        head = f.read(24)
    w, h = (int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")) if head[:4] == b"\x89PNG" else (0, 1)
    if w / h <= 1.5 or "split-screen" in action.lower() or not (creature / "creature.json").exists():
        return action
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_prompt", ROOT / "external/anymove/tools/make_prompt.py")
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    text = mp.prompt(creature, action).replace("One continuous 5-second shot", f"One continuous {seconds}-second shot")
    if re.search(r"slow[ -]?motion|slowly", action, re.I):  # the template forbids slow motion; this move asks for it
        text = text.replace("at natural speed, no slow motion", "in slow motion")
    return text


def wan_job(body):
    project = inside(OUT, body["project"])
    creature = inside(OUT, body["creature"])
    move, model = body["move"].strip(), body.get("model", WAN_MODELS[0])
    if not NAME.match(move) or model not in WAN_MODELS:
        raise ValueError("bad move name or model")
    ref = inside(OUT, body["ref"])
    prompt = body["prompt"].strip()
    if not prompt:
        raise ValueError("empty prompt")
    seconds, steps = int(body.get("seconds") or WAN_SECONDS[model]), int(body.get("steps") or 0)  # 0: the model's own
    if not 1 <= seconds <= WAN_SECONDS[model]:
        raise ValueError(f"{model} clips can be 1 to {WAN_SECONDS[model]} s long on this GPU")
    seeds = [int(s) for s in re.split(r"[,\s]+", str(body.get("seeds") or "").strip()) if s] or \
        [random.randrange(10**6) for _ in range(int(body.get("count") or 1))]
    if body.get("save_prompt"):
        save_prompt(creature, move, prompt)
    made = []
    for seed in seeds:
        out = creature / "runs/videos" / move / f"s{seed}"
        n = 2
        while (out / f"{model}.mp4").exists() or any(j["out"] == rel(out / f"{model}.mp4") and j["status"] in
                                                     ("queued", "running") for j in jobs):
            out, n = creature / "runs/videos" / move / f"s{seed}_{n}", n + 1
        out.mkdir(parents=True, exist_ok=True)
        clip = out / f"{model}.mp4"
        clip.with_suffix(".prompt.txt").write_text(framed(creature, prompt, seconds, ref) + "\n")  # what this clip was made
                                                                                                # from, whatever the move's
        clip.with_suffix(".job.json").write_text(json.dumps(       # prompt file says later
            {"model": model, "seed": seed, "seconds": seconds, "steps": steps or None, "ref": rel(ref),
             "fixes": body.get("fixes")}, indent=1))
        script = (f"source env.sh && cd external/anymove && exec {q(PY)} -m mocap.wan "
                  f"{q(ref, clip.with_suffix('.prompt.txt'), clip, seconds, seed, steps, model)}")
        made.append(submit("wan", f"{move} · {model} · seed {seed}", owner(project), script, out / "wan.log", clip))
    return made


def save_prompt(creature, move, prompt):
    """Write prompts/<move>.txt. Clips made from the text it replaces keep a copy of it first, so each clip still shows
    the prompt it was made from."""
    path = creature / "prompts" / f"{move}.txt"
    if path.exists():
        old = path.read_text().strip()
        if old == prompt:
            return
        for clip in (creature / "runs/videos" / move).rglob("*.mp4"):
            snap = clip.with_suffix(".prompt.txt")
            if not clip.name.endswith(("_h264.mp4", ".tmp.mp4")) and ".check" not in clip.parent.name and not snap.exists():
                snap.write_text(old + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt + "\n")


def creature_of(p):
    c = next((c for c in [p, *p.parents] if (c / "creature.json").exists()), None)
    if not c or OUT not in c.parents:
        raise ValueError(f"no creature folder above {p}")
    return c


def capture_job(body):
    """tools/capture.sh on a Wan clip; the capture remembers its clip and settings in source.json."""
    video = inside(OUT, body["video"])
    c = creature_of(video)
    name = body.get("name", "").strip() or video.parent.relative_to(c / "runs/videos").parts[0]
    if not NAME.match(name):
        raise ValueError("bad capture name")
    k = c / "runs/clips" / name
    if k.exists() and not body.get("replace"):
        base, n = name, 2
        while (c / "runs/clips" / name).exists():
            name, n = f"{base}_{n}", n + 1
        k = c / "runs/clips" / name
    ref = (read_json(video.with_suffix(".job.json")) or {}).get("ref")
    ref = OUT / ref if ref else c / "runs/ref/ref.png"
    solve, clean, speed = body.get("solve", ""), body.get("clean", ""), float(body.get("speed") or 1)
    body3d = bool(body.get("body3d")) and os.path.exists(BODY3D_PY)
    if not 0.25 <= speed <= 4:
        raise ValueError("speed is 0.25 to 4")
    k.mkdir(parents=True, exist_ok=True)
    (k / "source.json").write_text(json.dumps({"video": str(video), "ref": str(ref), "solve": solve, "clean": clean,
                                               "speed": speed, "body3d": body3d, "rig": rig_record(c)}, indent=1))
    b3d = f"BODY3D=1 BODY3D_PYTHON={q(BODY3D_PY)} " if body3d else ""
    ch = rig_of(c)[1]
    if ch:  # the one-project layout: SAM 3D Body's rotations retargeted onto rigging/rigged/ (anymove's solve options
        script = (f"source env.sh && CLIPS={q(c / 'runs/clips')} SPEED={speed:g} PYTHON={q(PY)} "  # don't apply)
                  f"rigging/capture.sh {q(ch, video, name)}")
        solve = clean = ""
        body3d = False
    else:
        script = (f"source env.sh && REF={q(ref)} SOLVE={q(solve)} CLEAN={q(clean)} SPEED={speed:g} PYTHON={q(PY)} {b3d}"
                  f"external/anymove/tools/capture.sh {q(c, video, name)}")
    extra = " ".join(x for x in (solve, clean, f"{speed:g}x" if speed != 1 else "", "SAM depth" if body3d else "") if x)
    return [submit("capture", f"capture · {name}" + (f" ({extra})" if extra else ""),
                   owner(c), script, k / "capture.log", k / "render/animation.glb")]


def reexport_job(body):
    """The capture's motion on the current rig again: renders, GLB and, when its clip is known, the showcase."""
    k = inside(OUT, body["capture"])
    c = creature_of(k)
    rd, ch = rig_of(c)
    cam = read_json(ch / "character.json" if ch else c / "creature.json", {}).get("camera", {})
    title = f"{read_json(c / 'creature.json', {}).get('name') or c.name} {k.name}"
    src = capture_info(k)["source"]
    source = read_json(k / "source.json", {})  # exported on the current rig from now on
    (k / "source.json").write_text(json.dumps({**source, "rig": rig_record(c)}, indent=1))
    script = (anymove(c) + f"{q(BLENDER)} -b -P blender/apply_motion.py -- "
              f"{q(rd / 'rigged.blend', rd / 'rig_info.json', k / 'motion.json', k / 'render')} --views persp,side "
              f"--samples 24 --cam {q(cam.get('cam', '-3.2,-3.0,1.6'))} --target {q(cam.get('target', '0,0,0.5'))} "
              f"--lens {q(cam.get('lens', 50))} --glb"
              + (f" --springs {q(rd / 'springs.json')}" if (rd / "springs.json").exists() else "")
              + f" > {q(k / 'blender.log')} 2>&1")
    if src and (k / "det.npz").exists():
        script += (f" && {q(PY)} -m mocap.showcase {q(OUT / src, k / 'det.npz', k / 'render', k / 'showcase.mp4', title)} && "
                   f"ffmpeg -loglevel error -y -i {q(k / 'showcase.mp4')} -c:v libx264 -crf 20 -pix_fmt yuv420p {q(k / 'showcase_h264.mp4')}")
    return [submit("reexport", f"export again · {k.name}", owner(k), script, k / "reexport.log",
                   k / "render/animation.glb")]


def rebuild_rig_job(body):
    """The rig built again, the old one kept beside it. In the one-project layout the reference frame (runs/ref) is
    rendered again from the new rig; in the old layout it stays as it is."""
    c = inside(OUT, body["creature"])
    if not (c / "creature.json").exists():
        raise ValueError("not a creature folder")
    rd, ch = rig_of(c)
    if ch:  # the one-project layout: rigging/rig.sh again over the same mesh, the old rig kept beside it
        keep = ch / time.strftime("rigged_before_%Y%m%d_%H%M%S")
        script = (f"source env.sh && cp -a {q(rd, keep)} && {capped(14)}bash rigging/rig.sh {q(ch)} && "
                  f"{capped(12)}bash rigging/ref_frame.sh {q(ch.parent)}")  # Wan's reference frame, from the new rig
        return [submit("rig", f"rebuild rig · {c.parent.name} (old one kept in {keep.name})", owner(c),
                       script, ch / "rebuild.log", rd / "rigged.blend")]
    keep = c / time.strftime("rigged_before_%Y%m%d_%H%M%S")
    script = (anymove(c) + f"cp -a {q(c / 'rigged', keep)} && {q(BLENDER)} -b -P blender/build_rig.py -- "
              f"{q(c / 'prep/static.blend', c / 'rig.json', c / 'rigged')}")
    if "hair" in read_json(c / "creature.json", {}):  # long hair: its own mesh and spring bones, as tools/rig.sh does
        script += f" && {q(BLENDER)} -b -P blender/add_hair.py -- {q(c / 'rigged/rigged.blend', c / 'creature.json')}"
    return [submit("rig", f"rebuild rig · {c.name} (old one kept in {keep.name})", owner(c), script,
                   c / "rigged/rebuild.log", c / "rigged/rigged.blend")]


def promote_job(body):
    """Make one of the project's TRELLIS models its character's mesh: rigging/promote.py preps it into anymove/ and
    rigs it into rigging/, the previous mesh, rig and captures moved to archive/<date-time>/ first."""
    project = inside(OUT, body["project"])
    model = inside(OUT, body["model"])
    if model.suffix != ".glb" or project not in model.parents:
        raise ValueError("pick a .glb model of this project")
    if set(model.relative_to(project).parts[:1]) & {"anymove", "rigging", "archive"}:
        raise ValueError("that is already a rigged or archived mesh: pick a TRELLIS run's model")
    busy = [j for j in jobs if j.get("status") in ("queued", "running") and
            any(str(j.get("out", "")).startswith(rel(project / d)) for d in ("anymove", "rigging"))]
    if busy:
        raise ValueError("a queued or running job writes to this project's rig: wait for it or cancel it first")
    height = body.get("height")
    script = (f"source env.sh && {capped(14)}{q(PY)} -m rigging.promote {q(project, model)}"
              + (f" --height {float(height)}" if height else ""))
    return [submit("rig", f"{owner(project)}: rig {model.relative_to(project)} (the old rig to archive/)", owner(project),
                   script, project / "promote.log", project / "rigging/rigged/rigged.blend")]


def paint_run(folder):
    """The TRELLIS run whose views can paint a model in `folder` (reference/reproject.py views_run, kept here as the
    studio imports no numpy): the folder, or an optimised copy's source run; None without views.json."""
    if (folder / "views.json").exists():
        return folder
    opt = read_json(folder / "optimise.json")
    src = (folder.parent.parent / opt["from"]).parent if opt and opt.get("from") else None
    return src if src and (src / "views.json").exists() else None


def paint_job(body):
    """A copy of a model painted with its run's views' pixels (reference/reproject.py), <name>_reprojected.glb beside
    it: for models made before TRELLIS runs did it themselves, or with "Paint from views" off."""
    name = character_id(body["project"])
    d = (OUT / name).resolve()
    model = inside(OUT, body["model"])
    if d not in model.parents or model.suffix != ".glb" or paint_run(model.parent) is None:
        raise ValueError("pick a TRELLIS model (or an optimised copy of one) made from views")
    out = model.with_name(f"{model.stem}_reprojected.glb")
    script = f"source env.sh && {capped(14)}{q(PY)} -m reference.reproject {q(model)}"
    return [submit("paint", f"{name}: paint {model.relative_to(d)} from its views", name, script,
                   model.with_name(f"{model.stem}_reprojected.log"), out)]


def retexture_job(body):
    """Another model's texture on the character's rig, without rigging again (rigging/retexture.py): a painted copy of
    the rig's own model (reference/reproject.py), or the plain model to put its texture back. Same mesh only."""
    project = inside(OUT, body["project"])
    model = inside(OUT, body["model"])
    if model.suffix != ".glb" or project not in model.parents:
        raise ValueError("pick a .glb model of this project")
    busy = [j for j in jobs if j.get("status") in ("queued", "running") and
            any(str(j.get("out", "")).startswith(rel(project / d)) for d in ("anymove", "rigging"))]
    if busy:
        raise ValueError("a queued or running job writes to this project's rig: wait for it or cancel it first")
    script = f"source env.sh && {capped(8)}{q(PY)} -m rigging.retexture {q(project, model)}"
    return [submit("rig", f"{owner(project)}: the rig wears {model.relative_to(project)}'s texture", owner(project),
                   script, project / "retexture.log", project / "rigging/promote.json", lane="cpu")]


def kimodo_job(body):
    """Captures of a move from words: rigging/kimodo_capture.py runs Kimodo, retargets each sample onto the project's
    rig and exports it, filed under the move (source.json names it). Kimodo reads its 16 GB text encoder (Llama 3 8B)
    into RAM before moving it to the GPU, so its cap is 20 GB, not 14: WSL peaks at about 20 GB while it loads."""
    project = inside(OUT, body["project"])
    move, prompt = body["move"].strip(), (body.get("prompt") or "").strip()
    use_recipe = bool(body.get("recipe"))
    if not NAME.match(move) or not (prompt or use_recipe):
        raise ValueError("a move name (letters, digits, - _ .) and a prompt")
    if use_recipe and not recipe_path(project, move).exists():
        raise ValueError("this move has no recipe yet: save one first")
    if not (project / "rigging/rigged/rigged.blend").exists():
        raise ValueError("this project has no rig yet: use a model for the rig first (Models tab)")
    duration, samples, seed = float(body.get("duration") or 2), int(body.get("samples") or 4), int(body.get("seed") or 0)
    if not (0.5 <= duration <= 10 and 1 <= samples <= 8):
        raise ValueError("0.5 to 10 seconds, 1 to 8 samples")
    if use_recipe:  # the prompt, length, samples and keyframes from the move's recipe
        script = (f"source env.sh && {capped(20)}{q(PY)} -m rigging.kimodo_capture {q(project, move)} --recipe "
                  f"--seed {seed}")
    else:
        script = (f"source env.sh && {capped(20)}{q(PY)} -m rigging.kimodo_capture {q(project, move, prompt)} "
                  f"--duration {duration} --samples {samples} --seed {seed}")
    clips = project / "anymove/runs/clips"
    clips.mkdir(parents=True, exist_ok=True)
    return [submit("kimodo", f"{move} · Kimodo{' recipe' if use_recipe else ''} · seed {seed}" + ("" if use_recipe else f" · {samples} samples"), owner(project), script,
                   clips / f"{move}_kimodo_s{seed}.log", clips / f"{move}_kimodo_s{seed}_0/render/animation.glb")]


def move_sources(args):
    """The other characters of creature's project with captures to copy (copy_moves_job): each capture's name, move
    and whether it is kept."""
    c = creature_of(inside(OUT, args["creature"]))
    me = c.parent
    out = []
    for o in sorted(me.parent.iterdir()):
        clips = o / "anymove/runs/clips"
        if o == me or not clips.is_dir():
            continue
        review = read_json(o / "review.json", {})
        caps = [{"name": k.name, "move": read_json(k / "source.json", {}).get("move"),
                 "kept": review.get(rel(k / "render/animation.glb"), {}).get("verdict") == "keep"}
                for k in sorted(clips.iterdir()) if (k / "motion.json").exists()]
        if caps:
            out.append({"character": rel(o), "creature": rel(o / "anymove"), "captures": caps})
    return {"sources": out}


def copy_moves_job(body):
    """Captures of another character of the project made again on this creature's rig (rigging/copy_moves.py): from
    Kimodo's own motion when the capture has it, else from the other rig's motion. Captures this creature
    already has are kept. The moves' prompts and Kimodo recipes come too, where this creature has none."""
    dst = creature_of(inside(OUT, body["creature"]))
    src = creature_of(inside(OUT, body["source"]))
    if src == dst or src.parent.parent != dst.parent.parent:
        raise ValueError("copy from another character of the same project")
    if not (rig_of(dst)[0] / "rigged.blend").exists():
        raise ValueError("this character has no rig yet: use a model for the rig first (Models tab)")
    names = body["captures"]
    if not names or not all(NAME.match(n) and (src / "runs/clips" / n / "motion.json").exists() for n in names):
        raise ValueError("pick captures of that character")
    (dst / "prompts").mkdir(exist_ok=True)
    for p in (src / "prompts").glob("*"):
        if p.is_file() and not (dst / "prompts" / p.name).exists():
            shutil.copy(p, dst / "prompts" / p.name)
    clips = dst / "runs/clips"
    clips.mkdir(parents=True, exist_ok=True)
    script = f"source env.sh && {capped(8)}{q(PY)} -m rigging.copy_moves {q(src.parent, dst.parent, *names)}"
    return [submit("copy_moves", f"copy {len(names)} move{'s' if len(names) > 1 else ''} from {src.parent.name}",
                   owner(dst), script, clips / f"copy_from_{src.parent.name}.log",
                   clips / names[-1] / "render/animation.glb", lane="cpu")]


# ---------------------------------------------------------------- items, loadouts and the game export

def is_item(name):
    parts = name.split("/")
    return len(parts) == 3 and parts[1] == ITEMS


def item_job(body):
    """One of an item's TRELLIS models chosen for the item (rigging/item_game.py): its grip estimated, and a light copy
    baked again from the run's voxels, aligned to a socket, for the preview and the game."""
    name = character_id(body["project"])
    if not is_item(name):
        raise ValueError("that is a character, not an item")
    d = (OUT / name).resolve()
    model = inside(OUT, body["model"])
    if d not in model.parents:
        raise ValueError("pick one of this item's models")
    if model.suffix != ".glb" or not (model.parent / "sample_xpu.pt").exists():
        raise ValueError("pick one of the item's TRELLIS models (its run keeps the voxels the light copy is baked from)")
    faces, tex = int(body.get("faces") or 5000), int(body.get("texture") or 1024)
    length = body.get("length_m")
    script = (f"source env.sh && PYTHONPATH={q(ROOT / 'external/trellis2/o-voxel')} {capped(14)}{q(PY)} "
              f"-m rigging.item_game {q(d, model)} --faces {faces} --texture {tex}"
              + (f" --length-m {float(length)} --length-from {q('set in the studio')}" if length else ""))
    return [submit("item", f"{name}: use {model.relative_to(d)} for the item", name, script, d / "item_game.log",
                   d / "game" / f"{d.name}.glb", lane="cpu")]


def project_items(group):
    """The items of a project that have a light copy: what can be equipped."""
    out = []
    for d in sorted((group / ITEMS).iterdir()) if (group / ITEMS).is_dir() else []:
        meta = read_json(d / "item.json") or {}
        if meta.get("game"):
            out.append({"name": d.name, "id": rel(d), "model": rel(d / meta["game"]["model"]),
                        "fits": meta.get("sockets", ["hand"]), "length_m": meta.get("length_m"),
                        "metres": meta["game"].get("metres", False)})
    return out


def equip_info(args):
    """For the equip preview: the character's rig (its sockets), its captures, its default loadout and the project's
    items."""
    c = creature_of(inside(OUT, args["creature"]))
    rd, ch = rig_of(c)
    info = read_json(rd / "rig_info.json", {})
    caps = [{"name": k.name, "glb": rel(k / "render/animation.glb"), "move": read_json(k / "source.json", {}).get("move")}
            for k in sorted((c / "runs/clips").iterdir()) if (k / "render/animation.glb").exists()] \
        if (c / "runs/clips").is_dir() else []
    return {"sockets": sorted(info.get("sockets", {})), "captures": caps,
            "loadout": read_json(c.parent / "rigging/loadout.json", {}), "items": project_items(c.parent.parent)}


def save_loadout(body):
    """A character's default loadout ({socket: item name}), for the export's manifest."""
    c = creature_of(inside(OUT, body["creature"]))
    loadout = {k: v for k, v in body["loadout"].items() if k.startswith("socket_") and v and NAME.match(v)}
    (c.parent / "rigging").mkdir(exist_ok=True)
    (c.parent / "rigging/loadout.json").write_text(json.dumps(loadout, indent=1))
    return {"loadout": loadout}


def export_sources(args):
    """What a project's game export can hold: each rigged character's captures (kept ones marked) and the items with
    a light copy."""
    group = inside(OUT, args["group"])
    if not is_group(group):
        raise ValueError("not a project")
    chars = []
    for c in sorted(group.iterdir()):
        clips = c / "anymove/runs/clips"
        if c.name in SKIP or c.name == ITEMS or not (c / "rigging/rigged/rigged.blend").exists() or not clips.is_dir():
            continue
        review = read_json(c / "review.json", {})
        caps = [{"name": k.name, "move": read_json(k / "source.json", {}).get("move"),
                 "kept": review.get(rel(k / "render/animation.glb"), {}).get("verdict") == "keep"}
                for k in sorted(clips.iterdir()) if (k / "motion.json").exists()]
        chars.append({"name": c.name, "captures": caps,
                      "sockets": sorted(read_json(c / "rigging/rigged/rig_info.json", {}).get("sockets", {}))})
    return {"characters": chars, "items": project_items(group),
            "export": read_json(group / "export/manifest.json")}


def export_job(body):
    """The project's game export (rigging/export_game.py) into <project>/export/, the previous one kept, and copied
    into a game project's folder if copy_to is given."""
    group = inside(OUT, body["group"])
    if not is_group(group):
        raise ValueError("not a project")
    chars = {n: caps for n, caps in body.get("characters", {}).items() if caps}
    for n, caps in chars.items():
        if not NAME.match(n) or not all(NAME.match(k) and (group / n / "anymove/runs/clips" / k / "motion.json").exists()
                                        for k in caps):
            raise ValueError(f"bad character or capture: {n}")
    items = [i for i in body.get("items", []) if NAME.match(i) and (group / ITEMS / i / "item.json").exists()]
    if not chars and not items:
        raise ValueError("pick captures or items to export")
    copy_to = (body.get("copy_to") or "").strip()
    if copy_to:  # a folder in a game project: export_game.py replaces it only if empty or an earlier copy of this export
        dest = pathlib.Path(copy_to).expanduser().resolve()
        if not dest.parent.is_dir():
            raise ValueError(f"{dest.parent} doesn't exist: the game folder's parent must")
        if dest == ROOT or ROOT in dest.parents or dest in ROOT.parents:
            raise ValueError("the copy goes into the game project, not this one")
        copy_to = str(dest)
    plan = group / ".export_plan.json"
    plan.write_text(json.dumps({"project": str(group), "characters": chars, "items": items,
                                **({"copy_to": copy_to} if copy_to else {})}, indent=1))
    script = f"source env.sh && {capped(8)}{q(PY)} -m rigging.export_game {q(plan)}"
    n = sum(len(v) for v in chars.values())
    return [submit("export", f"{group.name}: export for the game ({len(chars)} characters, {n} clips, {len(items)} items)",
                   group.name, script, group / "export.log", group / "export/manifest.json", lane="cpu")]


def optimise_job(body):
    """An optimised copy of a TRELLIS model (rigging/optimise.py): fewer faces, triangles or quads, its texture baked
    again from the run's voxels, as a model of its own beside the run (<run>_opt<k>k[_quads]/)."""
    name = character_id(body["project"])
    d = (OUT / name).resolve()
    model = inside(OUT, body["model"])
    if d not in model.parents or model.suffix != ".glb" or not (model.parent / "sample_xpu.pt").exists():
        raise ValueError("pick one of this character's TRELLIS models (its run keeps the voxels to bake again from)")
    faces, texture, mode = int(body.get("faces") or 30000), int(body.get("texture") or 2048), body.get("mode", "tris")
    if mode not in ("tris", "quads") or not 1000 <= faces <= 500000 or texture not in (512, 1024, 2048, 4096):
        raise ValueError("1,000 to 500,000 faces, tris or quads, a 512 to 4096 texture")
    run = model.parent
    base = run.parent / (f"{run.name}_opt{round(faces / 1000)}k" + ("_quads" if mode == "quads" else ""))
    out, n = base, 2
    while out.exists() or any(j.get("status") in ("queued", "running") and j.get("out", "").startswith(rel(out) + "/")
                              for j in jobs):
        out, n = base.with_name(f"{base.name}_{n}"), n + 1
    script = (f"source env.sh && PYTHONPATH={q(ROOT / 'external/trellis2/o-voxel')} {capped(14)}{q(PY)} -m rigging.optimise "
              f"{q(model)} --faces {faces} --mode {mode} --texture {texture} --out {q(out)}")
    return [submit("optimise", f"{name}: optimise {model.relative_to(d)} · {faces} faces · {mode} · {texture}", name,
                   script, out.parent / f"{out.name}.log", out / f"{d.name}_textured.glb", lane="cpu")]


def recipe_path(project, move):
    return project / "anymove/prompts" / f"{move}.kimodo.json"


def get_recipe(args):
    """A move's Kimodo recipe (rigging/kimodo_capture.py --recipe), and the captures it can take poses from."""
    project = inside(OUT, args["project"])
    if not NAME.match(args["move"]):
        raise ValueError("bad move name")
    rp = recipe_path(project, args["move"])
    clips = project / "anymove/runs/clips"
    sources = sorted(d.name for d in clips.glob("*") if (d / "kimodo.npz").exists()) if clips.is_dir() else []
    return {"text": rp.read_text() if rp.exists() else None, "path": str(rp.relative_to(OUT)), "sources": sources}


def save_recipe(body):
    """Write a move's recipe after checking it reads as one: a prompt, and pose sources that exist."""
    project = inside(OUT, body["project"])
    if not NAME.match(body["move"]):
        raise ValueError("bad move name")
    r = json.loads(body["text"])  # a JSON error goes back to the page as it is
    if not isinstance(r, dict) or not str(r.get("prompt", "")).strip():
        raise ValueError('a recipe needs a "prompt"')
    clips = project / "anymove/runs/clips"
    names = []
    jump = r.get("jump") or {}
    for q in r.get("poses", []) + [x for x in (jump.get("pose"), jump.get("crouch"), jump.get("release")) if x]:
        for k in ("from", "arms", "legs"):
            v = q.get(k) or []
            names += [v] if isinstance(v, str) else list(v)  # "arms" may be a list, applied in order
    base = lambda n: re.split(r"[:*]", str(n))[0]
    missing = sorted({base(n) for n in names if not (clips / base(n) / "kimodo.npz").exists()})
    if missing:
        raise ValueError("no such Kimodo capture: " + ", ".join(missing))
    rp = recipe_path(project, body["move"])
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(r, indent=1) + "\n")
    return get_recipe({"project": body["project"], "move": body["move"]})


def loop_job(body):
    """A capture made to loop (rigging/loop.py): the stretch whose end best matches its start, the seam closed, travel
    kept as root motion or taken out (in_place); exported as a capture of its own beside it, under the same move."""
    k = inside(OUT, body["capture"])
    if not (k / "motion.json").exists():
        raise ValueError("that capture has no motion.json")
    in_place = bool(body.get("in_place"))
    name = body.get("name", "").strip() or f"{k.name}_loop" + ("_in_place" if in_place else "")
    if not NAME.match(name):
        raise ValueError("bad capture name")
    script = (f"source env.sh && {capped(8)}{q(PY)} -m rigging.loop {q(k)} --name {q(name)}"
              + (" --in-place" if in_place else ""))
    out = k.parent / name
    return [submit("loop", f"loop · {name}", owner(k), script, k.parent / f"{name}.log",
                   out / "render/animation.glb", lane="cpu")]


def add_todo(body):
    c = inside(OUT, body["creature"])
    if not (c / "creature.json").exists():
        raise ValueError("not a creature folder")
    with open(c / "TODO.md", "a") as f:
        f.write(f"- [ ] {time.strftime('%Y-%m-%d')} {body['text'].strip()}\n")
    return {"todo": (c / "TODO.md").read_text()}


def capped(gb):
    """The start of a command held to gb of memory, with no swap to spill into: Qwen-Image-Edit once took all of WSL's
    memory and froze it, so Qwen and TRELLIS.2 runs fail on their own instead."""
    return f"systemd-run --user --scope -q -p MemoryMax={gb}G -p MemorySwapMax=0 "


def trellis_job(body):
    """TRELLIS.2 from one image (image), or from every view of a set (views, views_dir): example_xpu.py takes the views
    comma-separated and blends their predictions at each step, each voxel by the views that see it (blend
    "visibility", the default, for views named for their camera's angle) or by the plain mean (blend "mean").
    paint (default on): also <name>_textured_reprojected.glb, painted with the views' pixels (reference/reproject.py)."""
    name = character_id(body["project"])
    kind = body.get("pipeline", "512")
    folder = body.get("folder", "").strip() or kind + (f"_{body['views']}" if body.get("views") else "")
    if not NAME.match(folder) or kind not in TRELLIS_TYPES:
        raise ValueError("bad project, folder or pipeline type")
    used = None
    if body.get("views"):
        d, s = views_dir(body["views_dir"]), body["views"]
        if s not in SETS:
            raise ValueError(f"no view set {s}")
        missing = [v for v in SETS[s] if not (d / "views" / f"{v}.png").exists()]
        if missing:
            raise ValueError(f"views missing for {s}: {', '.join(missing)}")
        if "," in str(d):
            raise ValueError("the views folder's path has a comma, which example_xpu.py would read as two images")
        meta = read_json(d / "views/views.json", {})
        image = ",".join(str(d / "views" / f"{v}.png") for v in SETS[s])
        blend = "mean" if body.get("blend") == "mean" or any(v not in ANGLES and v != "front" for v in SETS[s]) else "visibility"
        used = {"set": s, "views_dir": rel(d), "versions": {v: meta.get(v, {}).get("version") for v in SETS[s]},
                "approved": {v: bool(meta.get(v, {}).get("approved")) for v in SETS[s]}, "blend": blend}
        if owner(d) != name:
            raise ValueError(f"those views belong to {owner(d)}, not {name}")
    else:
        image = inside(OUT, body["image"])
        if owner(image) != name:
            raise ValueError(f"that image belongs to {owner(image)}, not {name}")
        used = {"image": rel(image)}
    faces, tex = int(body.get("faces") or 200000), int(body.get("texture") or 2048)
    out = OUT / name / folder
    base, n = out, 2
    while out.exists():
        out, n = base.with_name(f"{base.name}_{n}"), n + 1
    out.mkdir(parents=True)
    if used:  # which versions of the views this model was made from, as they are replaced later
        (out / "views.json").write_text(json.dumps(used, indent=1))
    glb = out / f"{out.parent.name}_textured.glb"  # the character's own name, without its project
    paint = body.get("paint", True) is not False
    painted = glb.with_name(f"{glb.stem}_reprojected.glb")
    # paint: a second model beside the first, painted with the views' own pixels (reference/reproject.py); if that
    # fails the run still succeeds with its plain model. An item's light copy is made from the painted one if there
    # is one (rigging/item_game.py paints it too).
    script = (f"source env.sh && cd external/trellis2 && export PYTHONPATH=o-voxel && "
              + ("export TRELLIS_VIEW_WEIGHTS=mean && " if used.get("blend") == "mean" else "") +
              f"{capped(14)}{q(PY)} example_xpu.py {q(image, kind, out)} && "
              f"{capped(14)}{q(PY)} postprocess_xpu.py {q(out / 'sample_xpu.pt', glb, faces, tex)} && cd {q(ROOT)}"
              + (f" && {{ {capped(14)}{q(PY)} -m reference.reproject {q(glb)} || "
                 f"echo 'painting from the views failed: the model is kept as TRELLIS made it'; }}" if paint else "")
              + (f" && m={q(painted)} && {{ [ -f \"$m\" ] || m={q(glb)}; }} && "
                 f"PYTHONPATH={q(ROOT / 'external/trellis2/o-voxel')} {capped(14)}{q(PY)} "
                 f"-m rigging.item_game {q(OUT / name)} \"$m\" --if-new" if is_item(name) else ""))
    what = " · ".join(filter(None, [f"{used['set']} views" + (" · mean" if used["blend"] == "mean" else "") if used.get("set") else "",
                                     "painted" if paint else ""]))
    return [submit("trellis", f"{name}/{out.name} · {kind} · {faces // 1000}k faces" + (f" · {what}" if what else ""),
                   name, script, out / "run.log", glb)]


# ---------------------------------------------------------------- reference views (reference/views.py)

def views_dir(r):
    """The folder under outputs/ that holds views/ (any folder but outputs/ itself)."""
    d = inside(OUT, str(r).strip().strip("/"))
    if d == OUT.resolve() or not d.is_dir():
        raise ValueError(f"not a folder in outputs: {r}")
    return d


def writing(p):
    """The queued or running jobs that write inside p."""
    r = rel(p)
    return [j["label"] for j in jobs if j.get("status") in ("queued", "running") and (j["out"] == r or j["out"].startswith(r + "/"))]


def views_info(r):
    """A views folder for the page: each view's record from views.json with its image and older versions, and which
    views each set still lacks or has unapproved."""
    d = views_dir(r)
    v = d / "views"
    meta = read_json(v / "views.json", {})
    history = {}
    for p in sorted((v / "history").glob("*.v*.png")) if (v / "history").is_dir() else []:
        m = re.match(r"(.+)\.v(\d+)\.png$", p.name)
        if m:
            history.setdefault(m[1], []).append({"version": int(m[2]), "path": rel(p), "mtime": p.stat().st_mtime})
    views = {}
    for name in VIEWS:
        p = v / f"{name}.png"
        if p.exists() or name in meta or name in history:
            views[name] = {**meta.get(name, {}), "name": name, "path": rel(p) if p.exists() else None,
                           "mtime": p.stat().st_mtime if p.exists() else None,
                           "history": sorted(history.get(name, []), key=lambda x: -x["version"])}
    have = lambda x: x in views and views[x]["path"]
    sets = {s: {"views": vs, "missing": [x for x in vs if not have(x)],
                "unapproved": [x for x in vs if have(x) and not views[x].get("approved")]} for s, vs in SETS.items()}
    clips = [rel(p) for p in sorted((v / "turntable").glob("*.mp4")) if not p.name.endswith(("_h264.mp4", ".tmp.mp4"))] \
        if (v / "turntable").is_dir() else []
    return {"dir": rel(d), "views": views, "order": VIEWS, "angles": list(ANGLES), "sets": sets,
            "busy": writing(v), "turntable_clips": clips}


def views_py(d, *args):
    """reference/views.py on folder d in the Qwen Python, inside the memory cap (it loads Qwen-Image-Edit for --set,
    --make and --edit; the cap costs nothing for the rest)."""
    return f"{capped(12)}{q(QWEN_PY)} -m reference.views {q(d, *args)}"




def views_job(body):
    """Make the views a set lacks (set), or make these views again (make), with one Qwen load."""
    d = views_dir(body["views_dir"])
    v, seed = d / "views", int(body.get("seed") or 0)
    if not (v / "front.png").exists():
        raise ValueError("no front view yet: use an image as the front first")
    if body.get("set"):
        s = body["set"]
        if s not in SETS:
            raise ValueError(f"no view set {s}")
        todo = [x for x in SETS[s] if not (v / f"{x}.png").exists()]
        if any(x not in ANGLES for x in todo):
            raise ValueError("turntable views come from an orbit video: queue a turntable instead")
        if not todo:
            raise ValueError(f"the {s} set has all its views already")
        args, label, log = ("--set", s), f"views · {d.name} · {s}: make {', '.join(todo)}", v / f"set_{s}.log"
    else:
        todo = body.get("make") or []
        todo = todo.split() if isinstance(todo, str) else todo
        if not todo or any(x not in ANGLES for x in todo):
            raise ValueError(f"views Qwen can make: {', '.join(ANGLES)}")
        args, label, log = ("--make", *todo), f"views · {d.name} · remake {', '.join(todo)}", v / "make.log"
    out = v / f"{todo[0]}.png" if len(todo) == 1 else v
    script = f"source env.sh && exec {views_py(d, *args, '--seed', seed)}"
    return [submit("views", f"{label} · seed {seed}", owner(d), script, log, out)]


def views_edit_job(body):
    """Change one view with words (a Qwen edit of the view itself): a new version, the old one kept in history/."""
    d = views_dir(body["views_dir"])
    view, words, seed = body["view"], body.get("words", "").strip(), int(body.get("seed") or 0)
    if view not in VIEWS or not (d / "views" / f"{view}.png").exists():
        raise ValueError(f"no view {view} here")
    if not words:
        raise ValueError("say what to change")
    script = f"source env.sh && exec {views_py(d, '--edit', view, words, '--seed', seed)}"
    return [submit("views", f"views · {d.name} · edit {view} · seed {seed}", owner(d), script,
                   d / "views" / f"edit_{view}.log", d / "views" / f"{view}.png")]


def turntable_job(body):
    """An orbit video of the front view (Wan), and its frames as the turntable views, in one GPU job."""
    d = views_dir(body["views_dir"])
    v = d / "views"
    if not (v / "front.png").exists():
        raise ValueError("no front view yet: use an image as the front first")
    model, seed = WAN_MODELS[0], int(body.get("seed") or random.randrange(10**6))
    t = v / "turntable"
    t.mkdir(parents=True, exist_ok=True)
    clip, n = t / f"s{seed}.mp4", 2
    while clip.exists() or writing(clip):
        clip, n = t / f"s{seed}_{n}.mp4", n + 1
    prompt = clip.with_suffix(".prompt.txt")
    script = (f"source env.sh && {views_py(d, '--turntable-prompt', prompt)} && "
              f"(cd external/anymove && {q(PY)} -m mocap.wan {q(v / 'front.png', prompt, clip, WAN_SECONDS[model], seed, 0, model)}) && "
              f"{views_py(d, '--frames', clip, '--count', len(SETS['turntable']) - 1)}")
    return [submit("turntable", f"turntable · {d.name} · {model} · seed {seed}", owner(d), script,
                   t / "turntable.log", clip)]


def check_views_free(v):
    busy = writing(v)
    if busy:  # it rewrites views.json as each view is made, from what it read when it started
        raise ValueError(f"wait for {busy[0]} (or cancel it): it rewrites views.json, and would undo this")


def approve_view(body):
    d = views_dir(body["views_dir"])
    view = body["view"]
    check_views_free(d / "views")
    with _lock:
        vs = Views(d)
        if view not in VIEWS or not vs.path(view).exists():
            raise ValueError(f"no view {view} here")
        entry = vs.meta.setdefault(view, {})
        entry["approved"] = bool(body["approved"]) if "approved" in body else not entry.get("approved")
        vs.save()
    return views_info(rel(d))


def add_view(body):
    """Use an image under outputs/ (an upload, the project's image, an older version) as a view: a new version, as
    views.py --front / --add would make it. Other formats are turned into PNG, as the views are read as .png."""
    d = views_dir(body["views_dir"])
    view, image = body["view"], inside(OUT, body["image"])
    if view not in VIEWS:
        raise ValueError(f"unknown view {view}")
    if not image.is_file() or image.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError(f"not an image: {body['image']}")
    if owner(image) != owner(d):
        raise ValueError(f"that image belongs to {owner(image)}: use one of {owner(d)}'s")
    check_views_free(d / "views")
    with _lock:
        vs = Views(d)
        if image == vs.path(view).resolve():
            raise ValueError(f"that is the {view} view already")
        tmp = vs.dir / f".{view}.add.png"
        if image.suffix.lower() == ".png":
            shutil.copy(image, tmp)
        else:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(image), str(tmp)], check=True)
        vs.keep_old(view)
        tmp.rename(vs.path(view))
        vs.record(view, source=image.relative_to(ROOT).as_posix())
    return views_info(rel(d))


def free_name(parent, name):
    name = str(name or "").strip()
    if not NAME.match(name) or name in SKIP:
        raise ValueError("a name is letters, digits, _ . and -, starting with a letter or digit")
    if parent.is_dir() and any(p.name.lower() == name.lower() for p in parent.iterdir()):
        raise ValueError(f"{rel(parent / name) if parent != OUT else name} exists already")
    return name


def new_character(body):
    """A new character, in a project (project) or on its own: a folder that its images, views, models, rig and clips
    all go into, so working on it never touches another character. With item, a new item in the project's items/."""
    parent = OUT
    if body.get("group"):
        parent = OUT / str(body["group"]).strip()
        if not NAME.match(parent.name) or not is_group(parent):
            raise ValueError(f"no project {body['group']}")
    if body.get("item"):
        if parent == OUT:
            raise ValueError("an item belongs to a project, which every character in it can use")
        parent = parent / ITEMS
        parent.mkdir(exist_ok=True)
    elif parent != OUT and str(body.get("name", "")).strip().lower() == ITEMS:
        raise ValueError(f"{ITEMS} is the project's folder of items: name the character otherwise")
    name = free_name(parent, body.get("name"))
    (parent / name / "inputs").mkdir(parents=True)
    return {"name": owner(parent / name)}


def new_group(body):
    """A new project: a folder of characters, marked by its project.json."""
    name = free_name(OUT, body.get("name"))
    (OUT / name).mkdir()
    (OUT / name / "project.json").write_text(json.dumps({"name": name, "created": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    return {"name": name}


def upload(project, filename, data):
    project = character_id(project)
    if not NAME.match(filename) or pathlib.Path(filename).suffix.lower() not in (
            ".png", ".jpg", ".jpeg", ".webp"):
        raise ValueError("bad project or file name")
    d = OUT / project / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_bytes(data)
    return info(d / filename)


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if not self.path.startswith(("/files/", "/api/jobs")):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, p):
        size = p.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m[1] or m[2]):
            start, end = (int(m[1]), int(m[2]) if m[2] else size - 1) if m[1] else (size - int(m[2]), size - 1)
            end = min(end, size - 1)
        self.send_response(206 if m else 200)
        self.send_header("Content-Type", mimetypes.guess_type(p.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(p.stat().st_mtime))
        self.send_header("Cache-Control", "no-cache")
        if m:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(left, 1 << 20))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):  # the browser seeks by dropping the request
                    return
                left -= len(chunk)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n)

    def do_GET(self):
        url = urlparse(self.path)
        args = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                return self.send_file(PAGE)
            if url.path in ("/equip.js", "/export_example.html"):
                return self.send_file(PAGE.with_name(url.path[1:]))
            if url.path.startswith("/files/"):
                p = inside(OUT, unquote(url.path[len("/files/"):]))
                if p.suffix == ".mp4" and "raw" not in args:
                    p = playable(p)
                return self.send_file(p) if p.is_file() else self.send_json({"error": "not found"}, 404)
            if url.path == "/api/projects":
                chars, groups = listing()
                return self.send_json({"characters": chars, "groups": groups, "wan_models": WAN_MODELS, "wan_seconds": WAN_SECONDS, "trellis_types": TRELLIS_TYPES,
                                       "view_sets": SETS, "dry_run": DRY,
                                       "body3d": os.path.exists(BODY3D_PY), "body3d_python": BODY3D_PY,
                                       "root": str(ROOT), "distro": os.environ.get("WSL_DISTRO_NAME")})
            if url.path == "/api/project":
                return self.send_json(scan(args["name"]))
            if url.path == "/api/jobs":
                return self.send_json([job_view(j) for j in reversed(jobs[-80:])])
            if url.path == "/api/log":
                return self.send_json({"text": tail(args["path"], 20000)[0]})
            if url.path == "/api/views":
                return self.send_json(views_info(args["dir"]))
            if url.path == "/api/kimodo_recipe":
                return self.send_json(get_recipe(args))
            if url.path == "/api/move_sources":
                return self.send_json(move_sources(args))
            if url.path == "/api/equip":
                return self.send_json(equip_info(args))
            if url.path == "/api/export_sources":
                return self.send_json(export_sources(args))
            self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": str(e)}, 400)

    def do_POST(self):
        url = urlparse(self.path)
        args = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/upload":
                return self.send_json(upload(args["project"], args["name"], self.body()))
            body = json.loads(self.body() or b"{}")
            if url.path == "/api/characters":
                return self.send_json(new_character(body))
            if url.path == "/api/groups":
                return self.send_json(new_group(body))
            if url.path == "/api/review":
                return self.send_json(set_review(body))
            if url.path == "/api/reviews":
                return self.send_json(set_reviews(body))
            if url.path == "/api/todo":
                return self.send_json(add_todo(body))
            if url.path == "/api/delete":
                return self.send_json(delete(body))
            if url.path == "/api/views/approve":
                return self.send_json(approve_view(body))
            if url.path == "/api/views/add":
                return self.send_json(add_view(body))
            if url.path == "/api/kimodo_recipe":
                return self.send_json(save_recipe(body))
            if url.path == "/api/loadout":
                return self.send_json(save_loadout(body))
            makers = {"/api/jobs/wan": wan_job, "/api/jobs/trellis": trellis_job, "/api/jobs/capture": capture_job,
                      "/api/jobs/reexport": reexport_job, "/api/jobs/promote": promote_job, "/api/jobs/retexture": retexture_job, "/api/jobs/paint": paint_job, "/api/jobs/kimodo": kimodo_job, "/api/jobs/loop": loop_job, "/api/jobs/rebuild_rig": rebuild_rig_job, "/api/jobs/copy_moves": copy_moves_job, "/api/jobs/item": item_job, "/api/jobs/optimise": optimise_job, "/api/jobs/export": export_job,
                      "/api/jobs/views": views_job, "/api/jobs/views_edit": views_edit_job, "/api/jobs/turntable": turntable_job}
            if url.path in makers:
                return self.send_json([job_view(j) for j in makers[url.path](body)])
            if url.path == "/api/jobs/cancel":
                return self.send_json(job_view(cancel(body["id"])))
            if url.path == "/api/jobs/move_checks":
                return self.send_json(move_checks(body))
            if url.path == "/api/jobs/cancel_many":
                return self.send_json(cancel_many(body))
            self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": str(e)}, 400)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    OUT.mkdir(exist_ok=True)
    load_jobs()
    for lane in queues:
        threading.Thread(target=runner, args=(lane,), daemon=True).start()
    threading.Thread(target=transcoder, daemon=True).start()
    print(f"studio on http://localhost:{port}  (jobs run in {PY})", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    finally:  # Ctrl+C or kill: the jobs end with the server instead of running on unseen
        for j in jobs:
            if j["status"] == "running":
                stop_group(j)
