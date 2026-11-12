"""Find what went wrong in a Wan clip or a capture, when, on which limb, at which stage, and what would fix it.

python studio/checks/check.py CREATURE_DIR OUT.json [--det det.npz] [--motion motion.json] [--skin skin.json]
    [--rig-ok yes|no] [--find-source VIDEOS_DIR --overlay overlay.mp4]
Run with PYTHONPATH=external/anymove and MOCAP_CREATURE=CREATURE_DIR (for the frame layout).

Stages, from what each measures:
  video    both views lose a limb (the model smeared it), the two views show different moves, or the pose the video
           shows pushes a limb through the body: generate again
  reading  one view loses a limb the other sees clearly (it is hidden behind the body there): mocap.clean holds it;
           if the capture still looks wrong there, generate again with that limb kept in sight
  fit      the reader agrees with itself but the rig does not land on it: capture again with other solver settings
  skin     the pose is fine but the mesh collapses, or a limb sinks into the body because of how it is skinned
  rig      the animation was exported with skin weights that let a limb drag the body: export again with the rig
"""
import json, pathlib, sys

import numpy as np

args = sys.argv[1:]
creature_dir, out = pathlib.Path(args[0]), args[1]
opt = lambda k: args[args.index(k) + 1] if k in args else None
creature = json.load(open(creature_dir / "creature.json"))
# the rig: rigging/rigged/ beside anymove/ in the one-project layout (rigging/promote.py), else anymove's own rigged/
_ch = creature_dir.parent / "rigging"
info = json.load(open((_ch / "rigged" if (_ch / "character.json").exists() else creature_dir / "rigged") / "rig_info.json"))
noun = creature.get("noun") or creature.get("name") or "creature"

# limbs: the paint groups, named from creature.json's colour_code ("left arm pink, right arm blue, ...")
paint = creature.get("paint", {})
codes = [c.strip().rsplit(" ", 1)[0] for c in creature.get("colour_code", "").split(",") if c.strip()]
label = {k: (codes[i] if i < len(codes) and len(codes) == len(paint) else k) for i, k in enumerate(paint)}
bone_parent = {b["name"]: info["bones"][b["parent"]]["name"] if b["parent"] >= 0 else None for b in info["bones"]}
painted = {n: k for k, v in paint.items() for n in v["bones"]}


def limb_of(n):
    """The paint group of a bone or of the limb it hangs off (a blade off a hand)."""
    while n and n not in painted:
        n = bone_parent[n]
    return painted.get(n)


limb_of_bone = {n: limb_of(n) for n in bone_parent if limb_of(n)}
limb_of_joint = {b["tail_joint"]: limb_of_bone[b["name"]] for b in info["bones"] if b["name"] in limb_of_bone}
part = lambda bone: label[limb_of_bone[bone]] if bone in limb_of_bone else bone.split("_")[0].rstrip("0123456789")

THRESH = {"conf_low": 0.5, "conf_ok": 0.7, "gap_m": 0.12, "reproj_m": 0.12, "agree_m": 0.06,
          "squash": 0.03, "stretch": 0.08, "inside": 0.3}
issues, raw = [], {}


def windows(mask, min_len=2, bridge=5):
    """(start, end) frame runs of True, short gaps bridged, end exclusive."""
    out, t = [], 0
    idx = np.nonzero(mask)[0]
    for i in idx:
        if out and i - out[-1][1] <= bridge:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [(a, b) for a, b in out if b - a >= min_len]


def add(stage, limb, t0, t1, peak, what, fix, frames=None, alt=None, video_stage=None):
    """A limb hidden in one view is a note: every big gesture hides an arm from one camera for a while, and mocap.clean
    holds it. Anything else lasting 0.375 s or more (6 frames at 16 fps) is major. alt: a second fix to offer.
    video_stage: the stage the problem counts as in the Wan clip's own score (a capture can work around a clip's flaw)."""
    severity = "note" if stage == "reading" else "major" if t1 - t0 >= 0.375 else "minor"
    issues.append({"stage": stage, "limb": limb, "t0": round(t0, 3), "t1": round(t1, 3), "peak_t": round(peak, 3),
                   "what": what, "fix": fix, "severity": severity, **({"alt": alt} if alt else {}),
                   **({"video_stage": video_stage} if video_stage else {})})


def regenerate(phrase):
    return {"action": "regenerate", "label": "Generate again with this added to the prompt", "prompt_add": phrase}


det_path, motion_path, skin_path = opt("--det"), opt("--motion"), opt("--skin")
duration = 0.0
if det_path:
    from mocap import layout
    d = np.load(det_path)
    names = [str(n) for n in d["names"]]
    fps = float(d["fps"])
    T = len(d["front_conf"])
    duration = T / fps
    fc, sc = d["front_conf"], d["side_conf"]
    gap = np.abs(d["front_uv"][..., 1] - d["side_uv"][..., 1]) * layout.M_PER_PX
    reproj, trusted = None, np.zeros(fc.shape, bool)
    if motion_path:
        m = json.load(open(motion_path))
        for n, spans in m.get("trusted_front", {}).items():  # mocap.solve --trust-front: side reading dropped there
            for a, b in spans:
                trusted[a:b, names.index(n)] = True
        J = np.array(m["joints"])[:, [m["joint_names"].index(n) for n in names]]
        pf, ps = layout.project(J)
        rf = np.linalg.norm(pf - d["front_uv"], axis=-1) * layout.M_PER_PX
        rs = np.linalg.norm(ps - d["side_uv"], axis=-1) * layout.M_PER_PX
        reproj = np.maximum(np.where(fc > THRESH["conf_ok"], rf, 0), np.where((sc > THRESH["conf_ok"]) & ~trusted, rs, 0))
    raw["reader"] = {"conf_front": round(float(np.median(fc)), 3), "conf_side": round(float(np.median(sc)), 3),
                     "gap_m_p90": round(float(np.percentile(gap, 90)), 3)}
    # the whole clip: blank (the reader finds nothing), or the creature barely moves (how far the joints travel from
    # where they start; real moves travel 20 cm and more, a clip that only sways 2 cm)
    blank = np.median(fc) < 0.2 and np.median(sc) < 0.2
    travel = max(np.percentile(np.where(c > THRESH["conf_ok"], np.linalg.norm(uv - np.median(uv[:4], 0), axis=-1), 0).max(0), 90)
                 for uv, c in ((d["front_uv"], fc), (d["side_uv"], sc))) * layout.M_PER_PX
    raw["reader"]["travel_m"] = round(float(travel), 3)
    if blank:
        add("video", "whole clip", 0, duration, 0, f"the reader finds no {noun}: the clip is blank, or isn't the two-camera layout",
            {"action": "regenerate", "label": "Generate again with new seeds"})
    elif travel < 0.08:
        add("video", "whole body", 0, duration, duration / 2,
            f"the {noun} barely moves (no joint travels more than {travel * 100:.0f} cm): the clip doesn't perform the move",
            {"action": "regenerate", "label": "Generate again with new seeds"})
    # the side half should show the creature side on, its left and right halves (shoulders, hips) nearly in line; the
    # first frames are the reference, so they say how narrow. Wider than 0.8 of the front view's spread: the creature
    # turned toward the side camera, or the video model made the side half a second front view (prompts that say
    # "facing the camera" invite it). Either way the two halves no longer give the depth the capture needs.
    pairs = [(f"{p}_L", f"{p}_R") for p in ("shoulder", "hip") if f"{p}_L" in names and f"{p}_R" in names]
    if pairs and not blank:
        spread = lambda uv: sum(np.abs(uv[:, names.index(a), 0] - uv[:, names.index(b), 0]) for a, b in pairs)
        ratio = spread(d["side_uv"]) / np.maximum(spread(d["front_uv"]), 1)
        raw["reader"]["side_turn_frac"] = round(float(np.mean(ratio > 0.8)), 3)
        if np.median(ratio[:3]) < 0.4:  # the reference is side on (else the layout itself is something else)
            for a, b in windows(ratio > 0.8, min_len=round(0.75 * fps), bridge=3):  # a slash twists the torso briefly
                add("video", "whole body", a / fps, b / fps, (a + b) / 2 / fps,
                    f"the side half stops showing the {noun} side on: she turns toward the side camera, or the video model "
                    "turned that half into a second front view, so the two views no longer give depth",
                    regenerate(f"The right half shows the {noun} from the side, in profile, the whole time; in the right half "
                               "the body never turns toward the viewer."), b - a)
    for key in paint if not blank else ():
        js = [j for j, n in enumerate(names) if limb_of_joint.get(n) == key]
        if not js:
            continue
        lim = label[key]
        f_low, s_low = fc[:, js] < THRESH["conf_low"], sc[:, js] < THRESH["conf_low"]
        f_ok, s_ok = fc[:, js] > THRESH["conf_ok"], sc[:, js] > THRESH["conf_ok"]
        both = (f_low & s_low).any(1)
        one = ((f_low & s_ok) | (s_low & f_ok)).any(1) & ~both
        disagree = ((gap[:, js] > THRESH["gap_m"]) & f_ok & s_ok).any(1)
        for a, b in windows(both):
            add("video", lim, a / fps, b / fps, (a + b) / 2 / fps,
                f"the {lim} can't be read in either view: hidden behind the body from both cameras, or smeared by the video model",
                regenerate(f"The {noun}'s {lim} stays clearly visible and in one piece the whole time."), b - a)
        for a, b in windows(one, min_len=3):
            view = "front" if f_low[a:b].sum() > s_low[a:b].sum() else "side"
            add("reading", lim, a / fps, b / fps, (a + b) / 2 / fps,
                f"the {lim} is hidden in the {view} view (seen clearly in the other); the capture holds its last clear pose",
                regenerate(f"The {noun}'s {lim} stays in sight from both cameras and does not pass behind the body."), b - a)
        for a, b in windows(disagree, min_len=3):
            peak = a + int(gap[a:b][:, js].max(1).argmax())
            what = (f"the two views put the {lim} at different heights (up to {gap[a:b][:, js].max() * 100:.0f} cm): "
                    "the video's two halves don't show the same move")
            if trusted[a:b][:, js].any(1).mean() > 0.8:  # this capture already follows the front view there
                add("reading", lim, a / fps, b / fps, peak / fps, what + f"; this capture follows the front view, so the "
                    f"{lim}'s depth is a guess: check it from the side", None, b - a, video_stage="video")
                continue
            add("video", lim, a / fps, b / fps, peak / fps, what, {"action": "regenerate", "label": "Generate again with new seeds"}, b - a)
        if reproj is not None:
            agree = (gap[:, js] < THRESH["agree_m"]).all(1)
            off = (reproj[:, js] > THRESH["reproj_m"]).any(1) & agree
            for a, b in windows(off, min_len=3):
                add("fit", lim, a / fps, b / fps, (a + b) / 2 / fps,
                    f"the rig's {lim} misses where both views agree it is (by up to {reproj[a:b][:, js].max() * 100:.0f} cm): "
                    "a joint limit or the smoothing held it back",
                    {"action": "recapture", "label": "Capture again with lighter smoothing", "solve": "--acc 0.002"}, b - a)

if skin_path:
    s = json.load(open(skin_path))
    # the GLB's seconds; a capture played faster (capture.sh SPEED) maps back onto the clip's timeline
    speed = json.load(open(motion_path)).get("speed", 1) if motion_path else 1
    ts = np.array([r["t"] for r in s["frames"]]) * speed
    duration = max(duration, float(ts[-1]) if len(ts) else 0)
    sfps = s["fps"]
    if s["strays"]:
        worst = max(s["strays"], key=lambda r: r["stray"])
        fix = ({"action": "reexport", "label": "Export again with the current rig"} if opt("--rig-ok") == "yes" else
               {"action": "rebuild_rig", "label": "Rebuild the rig"})
        add("rig", part(worst["bone"]), 0, duration, 0,
            f"this animation's skin weights let the {part(worst['bone'])} drag {worst['stray']} vertices of the body with it"
            + (" (the current rig no longer does)" if opt("--rig-ok") == "yes" else ""), fix, 99)
    stray_bones = {r["bone"] for r in s["strays"]}
    for kind in ("inside", "squash", "stretch"):
        found = {}
        for i, r in enumerate(s["frames"]):
            if kind == "inside":
                for a, b, f in r["inside"]:
                    if f > THRESH["inside"]:
                        found.setdefault((a, b), {})[i] = f
            else:
                for bone, f in r[kind].items():
                    if f > THRESH[kind]:
                        found.setdefault((bone,), {})[i] = f
        for key, hits in found.items():
            mask = np.zeros(len(ts), bool)
            mask[list(hits)] = True
            for a, b in windows(mask, min_len=3):
                peak = max(range(a, b), key=lambda i: hits.get(i, 0))
                bone = key[0]
                if bone in stray_bones or any(bone_parent.get(x) == bone or bone_parent.get(bone) == x for x in stray_bones):
                    continue  # the stray weights above already explain it
                lim = part(bone)
                if kind == "inside":
                    other = part(key[1])
                    stage, what, fix = ("video", f"the {lim} passes into the {other} ({hits[peak]:.0%} of it)",
                                        regenerate(f"The {noun}'s {lim} stays clear of the {other}."))
                    if det_path and reproj is not None:  # did the fit put it there, or the video?
                        fr = slice(int(ts[a] * fps), max(int(ts[a] * fps) + 1, int(ts[b - 1] * fps)))
                        key_l = limb_of_bone.get(bone)
                        js = [j for j, n in enumerate(names) if limb_of_joint.get(n) == key_l]
                        if js and reproj[fr][:, js].max() > THRESH["reproj_m"]:
                            stage, what = "fit", what + ", where the rig misses the reader's joints"
                            fix = {"action": "recapture", "label": "Capture again with lighter smoothing", "solve": "--acc 0.002"}
                        else:
                            what += ": the video shows this pose"
                else:
                    stage = "skin"
                    what = f"the mesh around the {lim} ({bone}) {'collapses' if kind == 'squash' else 'stretches'} " \
                           f"({hits[peak]:.0%} of its edges)"
                    fix = {"action": "rig_task", "label": "Add to the rig to-do list (skin weights need hand work)"}
                add(stage, lim, ts[a], ts[b - 1] + 1 / sfps, ts[peak], what, fix, b - a)

# overlapping duplicates (two body parts both sinking into each other, a limb hidden in one view twice over): one
# problem per limb, stage and fix at a time, the first found standing for the rest
issues.sort(key=lambda i: (i["t0"], i["stage"]))
merged = []
for i in issues:
    same = next((j for j in merged if j["stage"] == i["stage"] and j["limb"] == i["limb"] and j["fix"] == i["fix"]
                 and i["t0"] <= j["t1"] and j["t0"] <= i["t1"]), None)
    if same:
        same["t1"] = max(same["t1"], i["t1"])
        if i["severity"] == "major" or same["t1"] - same["t0"] >= 0.375 and same["severity"] == "minor":
            same["severity"] = "major"
        continue
    merged.append(i)
for k, i in enumerate(merged):
    i["id"] = k

# a score to rank clips by: each stage's problems cost their share of the clip, weighted by how much they spoil the
# result (a hidden limb the capture holds costs little; a rig problem is fixed by exporting again)
WEIGHT = {"video": 1.0, "fit": 0.6, "skin": 0.6, "rig": 0.4, "reading": 0.1}


def score_of(stages, video=False):
    """video: score the Wan clip itself, counting what a capture worked around as the clip's own flaw."""
    cost = 0.0
    for stage in stages:
        spans = sorted((i["t0"], i["t1"]) for i in merged if ((i.get("video_stage") if video else None) or i["stage"]) == stage)
        covered, end = 0.0, 0.0
        for a, b in spans:  # time covered by this stage's problems, overlaps counted once
            a = max(a, end)
            if b > a:
                covered, end = covered + b - a, b
        cost += WEIGHT[stage] * covered / max(duration, 1e-6)
    return int(round(100 * max(0.0, 1 - cost)))


verdict = lambda score: "good" if score >= 80 else "minor" if score >= 50 else "bad"
score, video_score = score_of(WEIGHT), score_of(("video", "reading"), video=True)  # the second: the Wan clip's own share
result = {"duration": round(duration, 3), "limbs": [label[k] for k in paint], "issues": merged, "raw": raw, "score": score,
          "verdict": verdict(score), "video_score": video_score, "video_verdict": verdict(video_score)}

if opt("--find-source") and opt("--overlay") and det_path:  # which Wan clip a capture was made from
    import cv2
    from mocap.detect import read_frames
    target, _ = read_frames(opt("--overlay"))
    target = target[0].astype(np.float32)
    best = None
    for v in sorted(pathlib.Path(opt("--find-source")).rglob("*.mp4")):
        if v.name.endswith(("_h264.mp4", ".tmp.mp4")) or ".check" in v.parent.name:
            continue
        cap = cv2.VideoCapture(str(v))
        n, w, h = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        ok, f = cap.read()
        if not ok or n != T or w / h < 1.5:
            continue
        f = cv2.warpAffine(f[..., ::-1], d["calib"], (layout.W, layout.H)).astype(np.float32)
        err = float(np.abs(f - target).mean())
        if best is None or err < best[0]:
            best = (err, v)
    if best and best[0] < 12:
        result["source"] = str(best[1])
json.dump(result, open(out, "w"), indent=1)
print(f"{result['verdict']} (score {score}): {len(merged)} issues -> {out}")
for i in merged:
    print(f"  {i['t0']:5.2f}-{i['t1']:5.2f}s  {i['stage']:7s} {i['severity']:5s} {i['limb']:10s} {i['what']}")
