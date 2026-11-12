"""MakeHuman's CC0 base body (rigging/data/makehuman/): mesh, skeleton joints and skin weights, in Blender's axes.

No MakeHuman code is used, only its data files. The mesh is in decimetres with y up and the body facing +z; here it is
metres, z up, facing -y, standing on z = 0, its left at +x (as our characters).
"""
import fnmatch, json, pathlib

import numpy as np

DATA = pathlib.Path(__file__).parent / "data" / "makehuman"


def mirror(name, s):
    """A template name for side s: .L -> .R (MakeHuman), _L -> _R (ours), and either kind inside dicts."""
    if s == "L":
        return name
    if isinstance(name, dict):
        return {mirror(k, s): v for k, v in name.items()}
    for a, b in ((".L", ".R"), ("_L", "_R")):
        if name.endswith(a):
            return name[:-2] + b
    return name


def load(groups=("body", "helper-skirt"), data=DATA):
    """verts (N,3), faces {group: quads} for the named groups, joints {"bone.head"/"bone.tail": point},
    weights {bone: (idx, w)}."""
    v, faces = [], {g: [] for g in groups}
    cur = None
    for line in open(data / "base.obj"):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            cur = line.split()[1]
        elif line.startswith("f ") and cur in groups:
            faces[cur].append([int(t.split("/")[0]) - 1 for t in line.split()[1:]])
    v = np.array(v)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1) * 0.1
    skel = json.load(open(data / "default.mhskel"))
    faces = {g: np.array(f) for g, f in faces.items()}
    v[:, 2] -= v[np.unique(faces[groups[0]])][:, 2].min()  # the soles on the ground
    pos = {k: v[idx].mean(0) for k, idx in skel["joints"].items()}
    joints = {}
    for b, spec in skel["bones"].items():
        joints[b + ".head"], joints[b + ".tail"] = pos[spec["head"]], pos[spec["tail"]]
    raw = json.load(open(data / skel.get("weights_file", "default_weights.mhw")))["weights"]
    weights = {b: (np.array([i for i, _ in l], int), np.array([w for _, w in l])) for b, l in raw.items()}
    return v, faces, joints, weights


def grouped_weights(weights, rules, n, columns):
    """MakeHuman's weights summed into our bones: rules are (pattern, bone or {bone: share}), the first match wins.
    Returns (n, len(columns)) normalised per vertex."""
    W = np.zeros((n, len(columns)))
    col = {c: k for k, c in enumerate(columns)}
    for b, (idx, w) in weights.items():
        target = next(t for p, t in rules if fnmatch.fnmatchcase(b, p))
        for name, share in (target.items() if isinstance(target, dict) else [(target, 1.0)]):
            np.add.at(W[:, col[name]], idx, w * share)
    return W / W.sum(1, keepdims=True).clip(1e-9)


def rules_for(spec):
    """A template's "makehuman" map, both sides: [(pattern, target)], in order, left first."""
    out = []
    for p, t in spec["map"]:
        out.append((p, t))
        if p.endswith(".L"):
            out.append((mirror(p, "R"), mirror(t, "R")))
    return out
