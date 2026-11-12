"""Kimodo (NVIDIA's text-to-motion model) on the SOMA skeleton: per frame, its joints' world rotations and positions.

python -m rigging.kimodo_gen OUT_DIR "A person ..." [--duration 3] [--samples 4] [--seed 0] [--steps 100]
    [--model Kimodo-SOMA-RP-v1.2]
python -m rigging.kimodo_gen OUT_DIR --from MOTION.npz [MOTION.npz ...]

A compound move is a sequence of prompts separated by "|", each optionally starting with its length, played one after
another with Kimodo's own transitions: "0.6s: A person crosses their arms. | 1.2s: A person jumps and spins around. |
A person lands on one knee." Segments without a length share what --duration leaves.

OUT_DIR/kimodo_NN.npz, one per sample (or per --from motion, a Kimodo NPZ such as the demo's examples):
  rots   (T, J, 3, 3)  each joint's world rotation since SOMA's rest pose (at rest every one is the identity)
  joints (T, J, 3)     each joint's position, metres
  rest   (J, 3)        the rest pose's joint positions (a T-pose)
  names (J,), parents (J,), fps, foot_contacts (T, 4), text
in Kimodo's axes: y up, the character facing +z, x to its left. J is 77 (somaskel77, what generation writes) or 30.

Constraints, at a time in seconds ("end": the last frame), shared by every sample:
  --pose TIME=MOTION.npz[:FRAME]  the full body as in a frame of another Kimodo motion (one of these npz; FRAME defaults
                                  to its last), moved to stand on the start's spot (x = z = 0) facing the start's way
                                  (+z), or turned to face DEG with TIME=MOTION.npz:FRAME@DEG. A landing pose made on
                                  its own, then pinned at the end of a jump, so the jump lands right into it.
                                  ...+arms=OTHER.npz[:FRAME] takes both arms (shoulders down) from another frame,
                                  +legs=... both legs (hips down): one sample's body, another's arms or legs.
                                  +arms=OTHER.npz:FRAME*0.5 goes only that far from the arms so far (the frame's
                                  own, or an earlier +arms=) to OTHER's, joint by joint: arms half spread. Several
                                  +arms= apply in order; so do all the + steps (+legs=A+straight+legs=B*0.3:
                                  straightened, then a third of the way to B's).
                                  +straight straightens both knees (SOMA's rest pose has them straight).
                                  +upright stands the hips, spine, neck and head up straight (as at rest), facing
                                  the pose's way: no lean, whatever the source frame's.
                                  ...^Y (before any @DEG) puts the hips Y metres above the ground (y = 0): pins
                                  along a jump's arc make it as high and as long as the arc.
                                  ...>Z (after any @DEG) puts the hips Z metres ahead of the start's spot (along
                                  the way the start faces): a jump that travels.
  --heading TIME=DEG              where the hips face at that time (0: the start's way, 180: turned round), on the
                                  start's spot: a spin that has to happen, half way through it.
                                  TIME=DEG>Z: Z metres ahead of the start's spot (its +z) instead.
                                  TIME=DEG>Z,X: and X metres to its left (+x): a strafe facing one way while it
                                  travels sideways.

Needs a Python with kimodo importable (pip install -e ~/src/kimodo, see docs/rigging-plan.md) and, to generate,
access to the gated meta-llama/Meta-Llama-3-8B-Instruct on Hugging Face (its text encoder, LLM2Vec, is built on it).
Runs on CUDA or Intel's xpu: everything on the GPU takes about 17 GB. Kimodo is Apache 2.0, the SOMA weights NVIDIA's
Open Model License (commercial use allowed), Llama 3 Meta's licence; nothing of them is bundled.
"""
import argparse, json, os, re
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TEXT_ENCODER_MODE", "local")  # else it first looks for a text encoder server


def device():
    return "cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu"


def skeleton(n_joints):
    import kimodo
    from kimodo.skeleton.definitions import SOMASkeleton30, SOMASkeleton77
    cls = {30: SOMASkeleton30, 77: SOMASkeleton77}[n_joints]
    return cls(Path(kimodo.__file__).parent / "assets" / "skeletons" / cls.name)


def save(path, sk, rots, joints, fps, contacts, text):
    names = list(sk.bone_order_names)
    parents = np.array([-1 if p is None else names.index(p) for _, p in sk.bone_order_names_with_parents])
    np.savez(path, rots=rots, joints=joints, rest=sk.neutral_joints.detach().cpu().numpy(), names=np.array(names),
             parents=parents, fps=float(fps), foot_contacts=contacts, text=text)
    print(f"saved {path}: {len(joints)} frames at {fps} fps")


def segments(text, duration, fps):
    """[(prompt, frames)] from "prompt | 0.8s: prompt | ...": given lengths kept, the rest of duration shared."""
    parts = []
    for seg in (x.strip() for x in text.split("|")):
        m = re.match(r"^(\d+(?:\.\d+)?)\s*s\s*:\s*(.+)$", seg)
        parts.append((m[2], float(m[1])) if m else (seg, None))
    given = sum(d for _, d in parts if d)
    free = [i for i, (_, d) in enumerate(parts) if d is None]
    each = max(duration - given, 0.5 * len(free)) / len(free) if free else 0
    return [(t, int(round((d if d else each) * fps))) for t, d in parts]


def frame_at(t, n_frames, fps):
    return n_frames - 1 if t == "end" else min(int(round(float(t) * fps)), n_frames - 1)


def constraints(model, poses, headings, n_frames, dev):
    """Kimodo constraint sets from --pose and --heading (see the docstring)."""
    from kimodo.constraints import Root2DConstraintSet, load_constraints_lst
    from kimodo.geometry import axis_angle_to_matrix, matrix_to_axis_angle
    out = []
    sk77 = skeleton(77)
    names = list(sk77.bone_order_names)
    root = names.index("Hips")
    lhip, rhip = names.index("LeftLeg"), names.index("RightLeg")
    parents = [-1 if q is None else names.index(q) for _, q in sk77.bone_order_names_with_parents]

    def below(tops):  # these joints and everything below them
        out = []
        for i in range(len(names)):
            k = i
            while k >= 0 and names[k] not in tops:
                k = parents[k]
            if k >= 0:
                out.append(i)
        return out

    parts = {"arms": below(("LeftShoulder", "RightShoulder")), "legs": below(("LeftLeg", "RightLeg"))}

    rest = sk77.neutral_joints.detach().cpu().numpy().astype(np.float64)

    def yaw_to(d, deg):
        """The turn about y that brings the hip vector d (right hip - left hip) to Kimodo's heading deg."""
        turn = np.radians(deg) - np.arctan2(d[2], -d[0])  # Kimodo's heading: atan2(dz, -dx) of right - left hip
        for sign in (1, -1):  # whichever way round brings it there
            c, s_ = np.cos(sign * turn), np.sin(sign * turn)
            Ry = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
            d2 = Ry @ d
            if abs(np.angle(np.exp(1j * (np.arctan2(d2[2], -d2[0]) - np.radians(deg))))) < 1e-4:
                return Ry
        raise AssertionError("no turn about y reaches that heading")

    def load_frame(src):
        path, _, fr = src.partition(":")
        z = np.load(path)
        if z["rots"].shape[1] != 77:
            raise SystemExit(f"{path}: a pose needs a 77-joint Kimodo motion")
        f = int(fr) if fr else len(z["rots"]) - 1
        return z["rots"][f].astype(np.float64), z["joints"][f].astype(np.float64), path, f

    for spec in poses:
        spec, *extra = spec.split("+")
        steps = [x if x in ("straight", "upright") else x.split("=", 1) for x in extra]  # applied in this order
        if any(not isinstance(x, str) and (len(x) != 2 or x[0] not in parts) for x in steps):
            raise SystemExit(f"--pose: +arms=, +legs=, +straight or +upright, not {extra}")
        t, src = spec.split("=", 1)
        deg, height, fwd = 0.0, None, 0.0
        if ">" in src:
            src, fwd = src.rsplit(">", 1)
            fwd = float(fwd)
        if "@" in src:
            src, deg = src.rsplit("@", 1)
            deg = float(deg)
        if "^" in src:
            src, height = src.rsplit("^", 1)
            height = float(height)
        R, P, path, f = load_frame(src)
        Ry = yaw_to(P[rhip] - P[lhip], deg)
        Rg = Ry[None] @ R
        local = sk77.global_rots_to_local_rots(torch.tensor(Rg, dtype=torch.float32)[None])[0]
        for step in steps:
            if step == "upright":  # the root's rotation is its world one: only the turn that faces it the pose's way
                local[root] = torch.tensor(yaw_to(rest[rhip] - rest[lhip], deg), dtype=torch.float32)
                local[[names.index(k) for k in ("Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head")]] = torch.eye(3)
                path += ", upright"
                continue
            if step == "straight":
                local[[names.index("LeftShin"), names.index("RightShin")]] = torch.eye(3)
                path += ", knees straight"
                continue
            part, other = step
            mix = 1.0
            if "*" in other:
                other, mix = other.rsplit("*", 1)
                mix = float(mix)
            Ra, _, apath, af = load_frame(other)
            js = parts[part]
            them = sk77.global_rots_to_local_rots(torch.tensor(Ra, dtype=torch.float32)[None])[0][js]
            if mix < 1:  # part way from the pose's own (so far) to theirs, joint by joint
                now = local[js]
                local[js] = now @ axis_angle_to_matrix(mix * matrix_to_axis_angle(now.transpose(-1, -2) @ them))
            else:
                local[js] = them
            path = f"{path} with the {part} of frame {af} of {apath}" + (f" x{mix:g}" if mix < 1 else "")
        rootpos = Ry @ P[root]
        rootpos[[0, 2]] = 0, fwd
        if fwd:
            path += f", {fwd:.2f} m forward"
        if height is not None:
            rootpos[1] = height
            path += f", hips at {height:.2f} m"
        out += load_constraints_lst([{"type": "fullbody", "frame_indices": [frame_at(t, n_frames, model.fps)],
                                      "local_joints_rot": matrix_to_axis_angle(local)[None].tolist(),
                                      "root_positions": [rootpos.tolist()]}], model.skeleton, device=dev)
        print(f"pose at frame {frame_at(t, n_frames, model.fps)}: frame {f} of {path}, facing {deg:.0f} deg", flush=True)
    for spec in headings:
        t, deg = spec.split("=")
        deg, _, fwd = deg.partition(">")
        fwd, _, side = fwd.partition(",")  # "FWD,SIDE": SIDE metres to the left of the start (+x), e.g. a strafe
        a = np.radians(float(deg))
        k = frame_at(t, n_frames, model.fps)
        out.append(Root2DConstraintSet(model.skeleton, torch.tensor([k]), torch.tensor([[float(side or 0), float(fwd or 0)]]),
                                       global_root_heading=torch.tensor([[np.cos(a), np.sin(a)]], dtype=torch.float32)
                                       ).to(dev))
        print(f"heading at frame {k}: {float(deg):.0f} deg", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("text", nargs="?")
    ap.add_argument("--from", dest="src", nargs="+")
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--model", default="Kimodo-SOMA-RP-v1.2")
    ap.add_argument("--pose", action="append", default=[], help="TIME=MOTION.npz[:FRAME][@DEG]")
    ap.add_argument("--heading", action="append", default=[], help="TIME=DEG")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.src:  # motions Kimodo already wrote; the demo's examples keep their prompt in meta.json beside them
        for i, src in enumerate(a.src):
            z = np.load(src)
            meta = Path(src).with_name("meta.json")
            text = json.load(open(meta)).get("text", "") if meta.exists() else ""
            save(f"{a.out}/kimodo_{i:02d}.npz", skeleton(z["posed_joints"].shape[1]), z["global_rot_mats"],
                 z["posed_joints"], 30.0, z["foot_contacts"], text)
        return

    from kimodo.model import load_model
    from kimodo.tools import seed_everything

    dev = device()
    os.environ.setdefault("TEXT_ENCODER_DEVICE", dev)  # the wrapper picks only cuda or cpu by itself
    model = load_model(a.model, device=dev, default_family="Kimodo")
    seed_everything(a.seed)
    segs = segments(a.text, a.duration, model.fps)
    if len(segs) > 1:
        print("segments:", "; ".join(f"{n / model.fps:.2f} s: {t}" for t, n in segs), flush=True)
    cons = constraints(model, a.pose, a.heading, sum(n for _, n in segs), dev)
    if cons:  # Kimodo crops the constraints to each segment and fails on one left with no frames: drop those
        rep = model.motion_rep
        whole = rep.create_conditions_from_constraints
        rep.create_conditions_from_constraints = lambda lst, *x, **k: whole(
            [c for c in lst if not hasattr(c, "frame_indices") or len(c.frame_indices)], *x, **k)
    with torch.no_grad():
        out = model([t for t, _ in segs], [n for _, n in segs], constraint_lst=cons, num_denoising_steps=a.steps,
                    num_samples=a.samples, multi_prompt=True, post_processing=True, return_numpy=True)
    sk = model.skeleton
    sk = getattr(sk, "somaskel77", sk)  # the model runs on 30 joints; its output is on the full 77
    for i in range(a.samples):
        save(f"{a.out}/kimodo_{i:02d}.npz", sk, out["global_rot_mats"][i], out["posed_joints"][i], model.fps,
             out["foot_contacts"][i], a.text)


if __name__ == "__main__":
    main()
