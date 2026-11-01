"""Local review studio: look at the TRELLIS.2 models and the Wan clips in outputs/ before going further, mark the ones
to keep, and queue new runs on the GPU.

    python3 studio/server.py [port=8765]        # then open http://localhost:8765 (WSL forwards it to Windows)

Standard library only. Jobs run one at a time (there is one GPU) in $STUDIO_PYTHON (default ~/venvs/trellis2-xpu) with
env.sh sourced, and each writes its log next to its output. Verdicts and notes go to outputs/<project>/review.json.
Browsers can't play the MPEG-4 Part 2 clips Wan writes, so each gets an H.264 copy beside it (<name>_h264.mp4, as
before), made in the background when a project is opened.
"""
import json, mimetypes, os, pathlib, queue, random, re, shlex, signal, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
STATE = OUT / ".studio"  # jobs.json and uploads
PY = os.environ.get("STUDIO_PYTHON", str(pathlib.Path.home() / "venvs/trellis2-xpu/bin/python"))
PAGE = pathlib.Path(__file__).with_name("index.html")
SKIP = {"dataset", "__pycache__"}  # 2,200 training poses: nothing to review
PLAYABLE = {"h264", "vp9", "av1"}
WAN_MODELS = ["wan22", "wan22-14b"]
TRELLIS_TYPES = ["512", "1024", "1024_cascade", "1536_cascade"]
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

mimetypes.add_type("model/gltf-binary", ".glb")


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


# ---------------------------------------------------------------- videos the browser can play

_codecs, _locks, _lock = {}, {}, threading.Lock()
transcodes = queue.Queue()


def codec(p):
    key = (p, p.stat().st_mtime)
    if key not in _codecs:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                            "-of", "csv=p=0", str(p)], capture_output=True, text=True)
        _codecs[key] = r.stdout.strip()
    return _codecs[key]


def h264_copy(p):
    return p.with_name(p.stem + "_h264.mp4")


def playable(p):
    """p, or its H.264 copy, made now if missing or older than p."""
    if codec(p) in PLAYABLE:
        return p
    copy = h264_copy(p)
    with _lock:
        lock = _locks.setdefault(copy, threading.Lock())
    with lock:  # the browser asks for a clip several times at once
        if not copy.exists() or copy.stat().st_mtime < p.stat().st_mtime:
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


def scan(name):
    project = inside(OUT, name)
    crs = creatures(project)
    models, others, moves = [], [], {}

    def move(c, m):
        key = f"{rel(c)}:{m}"
        if key not in moves:
            prompt = c / "prompts" / f"{m}.txt"
            moves[key] = {"key": key, "creature": rel(c), "move": m, "clips": [], "captures": [],
                          "prompt": prompt.read_text().strip() if prompt.exists() else ""}
        return moves[key]

    for d, dirs, files in os.walk(project):
        dirs[:] = sorted(x for x in dirs if x not in SKIP and not x.startswith("."))
        d = pathlib.Path(d)
        for f in sorted(files):
            p = d / f
            if p.suffix not in (".glb", ".mp4") or f.endswith(".tmp.mp4"):
                continue
            if f.endswith("_h264.mp4") and p.with_name(f[:-9] + ".mp4").exists():  # shown as its original
                continue
            item = info(p)
            if p.suffix == ".mp4":
                item["log"] = log_for(p)
                if codec(p) not in PLAYABLE and not h264_copy(p).exists():
                    transcodes.put(p)
            c = next((c for c in crs if c in p.parents), None)
            where = p.relative_to(c / "runs").parts if c and (c / "runs") in p.parents else ()
            if len(where) >= 3 and where[0] in ("videos", "clips"):
                entry = move(c, where[1])
                item["label"] = "/".join(where[2:])[:-4]
                if where[0] == "videos":
                    snap = p.with_suffix(".prompt.txt")
                    item["prompt"] = snap.read_text().strip() if snap.exists() else None
                    meta = p.with_suffix(".job.json")
                    item["job"] = json.loads(meta.read_text()) if meta.exists() else None
                    entry["clips"].append(item)
                else:
                    entry["captures"].append(item)
            elif p.suffix == ".glb":
                item["group"] = p.parent.relative_to(project).as_posix()
                models.append(item)
            else:
                item["group"] = p.parent.relative_to(project).as_posix()
                others.append(item)
    for c in crs:  # moves with a prompt and no clips yet
        for t in sorted((c / "prompts").glob("*.txt")):
            move(c, t.stem)
    for m in moves.values():
        m["mtime"] = max([x["mtime"] for x in m["clips"] + m["captures"]], default=0)
    images = [info(p) for d in (project, project / "inputs") if d.is_dir()
              for p in sorted(d.iterdir()) if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
    refs = [info(p) for c in crs for p in sorted((c / "runs").glob("ref*/ref.png"))]
    review = project / "review.json"
    return {"name": name, "models": models, "videos": others, "images": images, "refs": refs,
            "creatures": [rel(c) for c in crs], "moves": sorted(moves.values(), key=lambda m: -m["mtime"]),
            "review": json.loads(review.read_text()) if review.exists() else {}}


def set_review(body):
    project = inside(OUT, body["project"])
    path = project / "review.json"
    with _lock:
        data = json.loads(path.read_text()) if path.exists() else {}
        key = body["path"]
        inside(OUT, key)
        entry = {k: body.get(k) for k in ("verdict", "note") if body.get(k)}
        if entry:
            data[key] = {**entry, "updated": time.time()}
        else:
            data.pop(key, None)
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
    return data


# ---------------------------------------------------------------- GPU jobs, one at a time

jobs, pending = [], queue.Queue()
_jobs_file = STATE / "jobs.json"


def save_jobs():
    STATE.mkdir(parents=True, exist_ok=True)
    _jobs_file.write_text(json.dumps([{k: v for k, v in j.items() if k != "proc"} for j in jobs], indent=1))


def load_jobs():
    if _jobs_file.exists():
        for j in json.loads(_jobs_file.read_text()):
            if j["status"] in ("queued", "running"):
                j["status"] = "interrupted"  # the server stopped before it finished
            jobs.append(j)


def submit(kind, label, project, script, log, out):
    job = {"id": f"{int(time.time() * 1000):x}", "kind": kind, "label": label, "project": project, "script": script,
           "log": rel(log), "out": rel(out), "status": "queued", "created": time.time()}
    jobs.append(job)
    save_jobs()
    pending.put(job)
    return job


def runner():
    while True:
        job = pending.get()
        if job["status"] != "queued":
            continue
        log = OUT / job["log"]
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "ab") as f:
            f.write(f"$ {job['script']}\n".encode())
            f.flush()
            proc = subprocess.Popen(["bash", "-c", job["script"]], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        job.update(status="running", started=time.time(), proc=proc)
        save_jobs()
        code = proc.wait()
        if job["status"] == "running":
            job["status"] = "done" if code == 0 else "failed"
        job.update(ended=time.time(), returncode=code)
        job.pop("proc", None)
        if job["status"] == "done" and job["kind"] == "wan":  # its H.264 copy, ready when the page next looks
            transcodes.put(OUT / job["out"])
        save_jobs()


def cancel(job_id):
    for j in jobs:
        if j["id"] == job_id and j["status"] in ("queued", "running"):
            if j["status"] == "running" and j.get("proc"):
                os.killpg(j["proc"].pid, signal.SIGTERM)
            j["status"] = "cancelled"
            save_jobs()
            return j
    raise ValueError(f"no queued or running job {job_id}")


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
    out = {k: v for k, v in j.items() if k not in ("proc", "script")}
    out["tail"], out["steps"] = tail(j["log"])
    return out


def q(*args):
    return " ".join(shlex.quote(str(a)) for a in args)


def wan_job(body):
    project = inside(OUT, body["project"])
    creature = inside(OUT, body["creature"])
    move, model = body["move"].strip(), body.get("model", "wan22")
    if not NAME.match(move) or model not in WAN_MODELS:
        raise ValueError("bad move name or model")
    ref = inside(OUT, body["ref"])
    prompt = body["prompt"].strip()
    if not prompt:
        raise ValueError("empty prompt")
    seconds, steps = int(body.get("seconds") or 5), int(body.get("steps") or 0)  # 0: the model's own step count
    seeds = [int(s) for s in re.split(r"[,\s]+", str(body.get("seeds") or "").strip()) if s] or [random.randrange(10**6)]
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
        clip.with_suffix(".prompt.txt").write_text(prompt + "\n")  # what this clip was made from, whatever the move's
        clip.with_suffix(".job.json").write_text(json.dumps(       # prompt file says later
            {"model": model, "seed": seed, "seconds": seconds, "steps": steps or None, "ref": rel(ref)}, indent=1))
        script = (f"source env.sh && cd external/anymove && exec {q(PY)} -m mocap.wan "
                  f"{q(ref, clip.with_suffix('.prompt.txt'), clip, seconds, seed, steps, model)}")
        made.append(submit("wan", f"{move} · {model} · seed {seed}", project.name, script, out / "wan.log", clip))
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
            if not clip.name.endswith(("_h264.mp4", ".tmp.mp4")) and not snap.exists():
                snap.write_text(old + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt + "\n")


def trellis_job(body):
    name = body["project"].strip()
    folder = body.get("folder", "").strip() or body.get("pipeline", "512")
    kind = body.get("pipeline", "512")
    if not NAME.match(name) or not NAME.match(folder) or kind not in TRELLIS_TYPES:
        raise ValueError("bad project, folder or pipeline type")
    image = inside(OUT, body["image"])
    faces, tex = int(body.get("faces") or 200000), int(body.get("texture") or 2048)
    out = OUT / name / folder
    base, n = out, 2
    while out.exists():
        out, n = base.with_name(f"{base.name}_{n}"), n + 1
    out.mkdir(parents=True)
    glb = out / f"{name}_textured.glb"
    script = (f"source env.sh && cd external/trellis2 && export PYTHONPATH=o-voxel && "
              f"{q(PY)} example_xpu.py {q(image, kind, out)} && "
              f"{q(PY)} postprocess_xpu.py {q(out / 'sample_xpu.pt', glb, faces, tex)}")
    return [submit("trellis", f"{name}/{out.name} · {kind} · {faces // 1000}k faces", name, script, out / "run.log", glb)]


def upload(project, filename, data):
    if not NAME.match(project) or not NAME.match(filename) or pathlib.Path(filename).suffix.lower() not in (
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
            if url.path.startswith("/files/"):
                p = inside(OUT, unquote(url.path[len("/files/"):]))
                if p.suffix == ".mp4" and "raw" not in args:
                    p = playable(p)
                return self.send_file(p) if p.is_file() else self.send_json({"error": "not found"}, 404)
            if url.path == "/api/projects":
                names = sorted(p.name for p in OUT.iterdir() if p.is_dir() and not p.name.startswith("."))
                return self.send_json({"projects": names, "wan_models": WAN_MODELS, "trellis_types": TRELLIS_TYPES,
                                      "root": str(ROOT), "distro": os.environ.get("WSL_DISTRO_NAME")})
            if url.path == "/api/project":
                return self.send_json(scan(args["name"]))
            if url.path == "/api/jobs":
                return self.send_json([job_view(j) for j in reversed(jobs[-50:])])
            if url.path == "/api/log":
                return self.send_json({"text": tail(args["path"], 20000)[0]})
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
            if url.path == "/api/review":
                return self.send_json(set_review(body))
            if url.path == "/api/jobs/wan":
                return self.send_json([job_view(j) for j in wan_job(body)])
            if url.path == "/api/jobs/trellis":
                return self.send_json([job_view(j) for j in trellis_job(body)])
            if url.path == "/api/jobs/cancel":
                return self.send_json(job_view(cancel(body["id"])))
            self.send_json({"error": "not found"}, 404)
        except Exception as e:
            self.send_json({"error": str(e)}, 400)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    OUT.mkdir(exist_ok=True)
    load_jobs()
    threading.Thread(target=runner, daemon=True).start()
    threading.Thread(target=transcoder, daemon=True).start()
    print(f"studio on http://localhost:{port}  (jobs run in {PY})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
