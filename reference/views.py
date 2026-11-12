"""Views of one character or creature for TRELLIS.2, made from its front image, kept and versioned in DIR/views/.

    python -m reference.views DIR --front IMAGE                 # the front view everything starts from (copied in)
    python -m reference.views DIR --add back MY_BACK.png        # any view from an image of your own
    python -m reference.views DIR --set four [--seed 0]         # make the views a set lacks (one Qwen model load)
    python -m reference.views DIR --make back left [--seed 3]   # make (or remake) these views
    python -m reference.views DIR --edit front "words" [--seed 0]   # change any view with words; a new version
    python -m reference.views DIR --turntable-prompt FILE       # the prompt for an orbit video of the front (Wan)
    python -m reference.views DIR --frames CLIP.mp4 [--count 7] # that video's frames as views turntable_1..N
    python -m reference.views DIR --paths SET                   # the set's images, comma-separated, for example_xpu.py

Sets: front, front_34, front_side, four (front, left, back, right), both_34, eight (every 45 degrees), turntable.
A view is named for where the camera stands, going round the subject: front, front_left (45 degrees to the subject's
own left), left, back_left, back, back_right, right, front_right. Each made view is a Qwen-Image-Edit of the front
(reference/qwen.py); the prompts say nothing about any one character, so they suit any character or creature.
DIR/views/views.json keeps each view's prompt, seed, source, version and whether it is approved; older versions go to
DIR/views/history/. Views are worth reviewing before a TRELLIS run: one with a changed pose or design spoils the mesh.
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

ANGLES = {"front_left": 45, "left": 90, "back_left": 135, "back": 180, "back_right": 225, "right": 270, "front_right": 315}
SETS = {
    "front": ["front"],
    "front_34": ["front", "front_right"],
    "front_side": ["front", "left"],
    "four": ["front", "left", "back", "right"],
    "both_34": ["front", "front_left", "front_right"],
    "eight": ["front", "front_left", "left", "back_left", "back", "back_right", "right", "front_right"],
    "turntable": ["front"] + [f"turntable_{i}" for i in range(1, 8)],
}
KEEP = ("Keep everything about the subject exactly the same: its identity, face, hair or fur, body shape, proportions, "
        "clothing, colours, materials, accessories and anything it holds, and exactly the same pose.")
FRAME = "Full body, nothing cut off, plain light grey background, even studio lighting, no shadows on the ground."
TURNTABLE = ("The camera orbits smoothly all the way around the subject at a constant speed and height, while the "
             "subject stands perfectly still in the same pose and does not move at all. " + FRAME)


def where(view):
    a = ANGLES[view]
    side = "left" if a < 180 else "right"
    turn = a if a <= 180 else 360 - a
    # said as the picture shows it (which way the subject faces in the frame, which of its sides we see): "the camera
    # moved to the subject's own left" is easy to get backwards. With the camera on the subject's left, it faces the
    # picture's left.
    if a == 180:
        return "seen from directly behind: the subject faces straight away from the viewer and we see its back"
    if turn == 90:
        return (f"seen exactly side-on, in profile, facing the {side} edge of the picture, so that we see its own "
                f"{side} side")
    if turn == 45:
        return (f"seen from a three-quarter front view: turned 45 degrees so that it faces towards the {side} of the "
                f"picture and we see more of its own {side} side")
    return (f"seen from a three-quarter back view: turned away from the viewer, towards the {side} of the picture, so "
            f"that we see its back and its own {side} side")


def prompt_for(view):
    return f"Show the exact same subject {where(view)}. {KEEP} {FRAME}"


class Views:
    def __init__(self, d):
        self.dir = Path(d) / "views"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "views.json"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else {}

    def path(self, view):
        return self.dir / f"{view}.png"

    def save(self):
        self.meta_path.write_text(json.dumps(self.meta, indent=1))

    def record(self, view, **info):
        old = self.meta.get(view, {})
        version = old.get("version", 0) + 1
        self.meta[view] = {**info, "version": version, "approved": False, "made": time.strftime("%Y-%m-%d %H:%M")}
        self.save()

    def keep_old(self, view):
        """Move the current image of a view into history/ before it is replaced."""
        p = self.path(view)
        if p.exists():
            (self.dir / "history").mkdir(exist_ok=True)
            shutil.move(p, self.dir / "history" / f"{view}.v{self.meta.get(view, {}).get('version', 0)}.png")

    def missing(self, name):
        return [v for v in SETS[name] if not self.path(v).exists()]


def make(views, tasks):
    """tasks: (view, source view, prompt, seed). One Qwen load for all of them; each result becomes a new version."""
    from PIL import Image
    from reference.qwen import edit
    tmp = [(Image.open(views.path(src)), prompt, seed, views.dir / f".{view}.new.png") for view, src, prompt, seed in tasks]
    edit(tmp, log=lambda s: print(s, flush=True))
    for (view, src, prompt, seed), (_, _, _, new) in zip(tasks, tmp):
        views.keep_old(view)
        new.rename(views.path(view))
        views.record(view, prompt=prompt, seed=seed, source=src)
        print(f"view {view}: {views.path(view)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--front")
    ap.add_argument("--add", nargs=2, metavar=("VIEW", "IMAGE"))
    ap.add_argument("--set", choices=SETS)
    ap.add_argument("--make", nargs="+", choices=list(ANGLES))
    ap.add_argument("--edit", nargs=2, metavar=("VIEW", "WORDS"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turntable-prompt")
    ap.add_argument("--frames")
    ap.add_argument("--count", type=int, default=7)
    ap.add_argument("--paths", choices=SETS)
    a = ap.parse_args()
    views = Views(a.dir)

    if a.front:
        views.keep_old("front")
        shutil.copy(a.front, views.path("front"))
        views.record("front", source=str(a.front))
        print(f"view front: {views.path('front')}")
    if a.add:
        view, image = a.add
        if view not in SETS["eight"] + SETS["turntable"]:
            sys.exit(f"unknown view {view}")
        views.keep_old(view)
        shutil.copy(image, views.path(view))
        views.record(view, source=str(image))
        print(f"view {view}: {views.path(view)}")
    if a.set or a.make:
        todo = a.make or [v for v in views.missing(a.set) if v in ANGLES]
        if not views.path("front").exists():
            sys.exit("no front view: give --front IMAGE first")
        if a.set == "turntable" and views.missing("turntable"):
            sys.exit("turntable views come from an orbit video: --turntable-prompt, then Wan, then --frames")
        if todo:
            make(views, [(v, "front", prompt_for(v), a.seed) for v in todo])
        else:
            print("nothing to make")
    if a.edit:
        view, words = a.edit
        if not views.path(view).exists():
            sys.exit(f"no view {view}")
        make(views, [(view, view, words, a.seed)])
    if a.turntable_prompt:
        Path(a.turntable_prompt).write_text(TURNTABLE + "\n")
    if a.frames:  # an orbit can't be trusted to reach any given angle, and TRELLIS needs none: evenly spaced frames
        n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames", "-of", "csv=p=0", a.frames], capture_output=True, text=True,
                               check=True).stdout.strip())
        for i in range(1, a.count + 1):
            view, k = f"turntable_{i}", round(i * (n - 1) / a.count)
            views.keep_old(view)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.frames, "-vf", f"select=eq(n\\,{k})", "-frames:v", "1",
                            str(views.path(view))], check=True)
            views.record(view, source=str(a.frames), frame=k)
            print(f"view {view}: frame {k} of {n}")
    if a.paths:
        if views.missing(a.paths):
            sys.exit(f"views missing for {a.paths}: {' '.join(views.missing(a.paths))}")
        print(",".join(str(views.path(v)) for v in SETS[a.paths]))


if __name__ == "__main__":
    main()
