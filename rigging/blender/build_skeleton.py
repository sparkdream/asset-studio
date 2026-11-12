"""Build a character's skeleton from a template placed on SAM 3D Body's joints, skin it, and save the rig.

blender -b static.blend -P rigging/blender/build_skeleton.py -- TEMPLATE.json MHR_REST.json CHARACTER.json OUT_DIR

TEMPLATE.json: rigging/templates/humanoid.json (bones on MHR joints, twist bones, the retarget's map).
MHR_REST.json: rigging.fit's joints, triangulated from the rest renders. CHARACTER.json: held props, hair, layout.
static.blend: the unrigged character, its mesh object named "body".

Writes OUT_DIR/rigged.blend, OUT_DIR/rig_info.json (bones, rest matrices, keypoints: what the retarget, renderer and
studio read), OUT_DIR/retarget_map.json (the template's map with both sides and the props) and OUT_DIR/joints_*.png.

Steps:
- joints from MHR, and the limb joints the template names (elbows, wrists, knees, ankles) moved to the middle of
  the mesh around them: rays cast across the limb from the joint, out to the surface;
- held props re-gripped in the fist (grip_deg) and fixed to the hand;
- weights, with the template's "makehuman" section (unless CHARACTER.json says "weights": "heat"): MakeHuman's CC0
  base body posed onto the skeleton, each vertex taking the weights of the nearest point of its body or skirt; the
  spine's share split along the spine bones; clothing hanging away from the body smoothed from what sits on it;
- else Blender's automatic weights on the main bones, then: smoothed; limbs pulled back from far-off vertices (an arm
  against a long coat); each limb's vertices kept off the other side's limb; split sharply along each limb by where a
  vertex sits on it;
- then twist bones given the part of their limb nearest their end; at most 4 bones per vertex; loose pieces moving
  with the surface they sit on.
"""
import json, math, pathlib, sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from rigging import makehuman  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
tmpl_path, mhr_path, char_path, out = args[:4]
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)
tmpl = json.load(open(tmpl_path))
char = json.load(open(char_path))
M = {k: np.array(v) for k, v in json.load(open(mhr_path))["joints"].items()}


def side(name, s):
    """A template name for side s: _L -> _R and l_ -> r_ for the right."""
    if s == "L":
        return name
    if isinstance(name, str):
        return name[:-2] + "_R" if name.endswith("_L") else ("r_" + name[2:] if name.startswith("l_") else name)
    if isinstance(name, list):
        return [side(x, s) for x in name]
    if isinstance(name, dict):
        return {k: side(v, s) for k, v in name.items()}
    return name


bones = []  # each side of each template bone, the left first
for b in tmpl["bones"]:
    for s in (("L", "R") if b["name"].endswith("_L") else ("L",)):
        bones.append({k: side(v, s) for k, v in b.items()})
names = [b["name"] for b in bones]
by = {b["name"]: b for b in bones}


def anchor(a):
    if isinstance(a, str):
        return M[a]
    if isinstance(a, list):
        return M[a[0]] + (M[a[1]] - M[a[0]]) * a[2]
    a0, a1, t = a["extend"]
    return M[a1] + (M[a1] - M[a0]) * t


body = bpy.data.objects["body"]
# generated meshes split vertices along texture seams: merge them, or the weights stop at each seam
bpy.context.view_layer.objects.active = body
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
n0 = len(body.data.vertices)
bpy.ops.mesh.remove_doubles(threshold=1e-5)
bpy.ops.object.mode_set(mode="OBJECT")
print("merged seam vertices:", n0, "->", len(body.data.vertices))

bvh = BVHTree.FromObject(body, bpy.context.evaluated_depsgraph_get())


def centre_in(p, d, reach=0.15):
    """The middle of the mesh around p, across direction d (rays cast out to the surface), or None outside it."""
    e1 = np.cross(d, [0, 0, 1] if abs(d[2]) < 0.9 else [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    for _ in range(3):
        hits = []
        for a in np.linspace(0, 2 * math.pi, 24, endpoint=False):
            loc, _, _, _ = bvh.ray_cast(Vector(p), Vector(math.cos(a) * e1 + math.sin(a) * e2), reach)
            if loc is not None:
                hits.append(np.array(loc))
        if len(hits) < 20:
            return None
        q = np.mean(hits, 0)
        p = p + (q - p) - ((q - p) @ d) * d
    return p


E0 = np.array([e.vertices[:] for e in body.data.edges])
co0 = np.array([v.co[:] for v in body.data.vertices])


def parting(a, b, lo, hi):
    """The height where the mesh parts two limbs: the highest cut below which the vertices nearest points a and b (two
    knees) are no longer joined, searched between lo and hi."""
    seeds = [np.argsort(np.linalg.norm(co0 - p, axis=1))[:20] for p in (a, b)]

    def joined(h):
        on = np.zeros(len(co0), bool)
        on[seeds[0]] = True
        below = co0[:, 2] < h
        while True:
            grow = below[E0[:, 0]] & below[E0[:, 1]] & (on[E0[:, 0]] != on[E0[:, 1]])
            if not grow.any():
                return on[seeds[1]].any()
            on[E0[grow].ravel()] = True

    for _ in range(9):
        h = (lo + hi) / 2
        lo, hi = (lo, h) if joined(h) else (h, hi)
    return lo


# The hip joints sit a set fraction of the height above the crotch, where the mesh parts the legs. SAM's hips were 16 cm
# above this character's crotch, and a leg swinging about a pivot that high drags the crotch into a sheet between the
# thighs (a body's hip joints are about 5% of its height above its crotch).
crotch_z = None
if "crotch" in tmpl:
    cr = tmpl["crotch"]
    a, b = M[cr["knees"][0]], M[cr["knees"][1]]
    crotch_z = parting(a, b, max(a[2], b[2]), M[cr["hips"][0]][2] + 0.05)
    height = M[cr["top"]][2] - min(M[cr["ground"][0]][2], M[cr["ground"][1]][2])
    for j in cr["hips"]:
        z = crotch_z + cr["above"] * height
        print(f"{j}: {(z - M[j][2]) * 100:+.1f} cm to {cr['above']:.0%} of the height above the crotch ({crotch_z:.3f} m)")
        M[j] = np.array([M[j][0], M[j][1], z])

# A limb's root (hip, shoulder) sits inside the body, where no ray finds the limb's own surface, and MHR's convention
# for it needn't match the mesh (its hip joints are 10 cm apart where this kind of body's thighs are 18): it goes on
# the limb's own axis, through the middle of the limb at two points along it, at the height of SAM's joint.
for kp, (bn, t1, t2) in [(side(k, s), side(v, s)) for k, v in tmpl.get("axis", {}).items() for s in ("L", "R")]:
    b = by[bn]
    if not isinstance(b["head"], str):
        continue
    h, t = anchor(b["head"]), anchor(b["tail"])
    d = (t - h) / np.linalg.norm(t - h)
    c1, c2 = centre_in(h + (t - h) * t1, d), centre_in(h + (t - h) * t2, d)
    if c1 is None or c2 is None:
        print(f"{kp}: no limb axis found, left where SAM put it")
        continue
    ax = (c1 - c2) / np.linalg.norm(c1 - c2)
    p = c1 + ax * (h[2] - c1[2]) / ax[2] if abs(ax[2]) > 0.3 else c1 + ax * ((h - c1) @ ax)
    shift = p - h
    if np.linalg.norm(shift) > 0.08:
        p = h + shift / np.linalg.norm(shift) * 0.08
    M[b["head"]] = p
    print(f"{kp}: on the limb's axis, moved {np.linalg.norm(p - h) * 100:.1f} cm")

# limb joints to the middle of the mesh around them
for kp in [side(k, s) for k in tmpl.get("centre", []) for s in ("L", "R")]:
    child = next((b for b in bones if b.get("head_kp") == kp), None)
    if child is None or not isinstance(child["head"], str):
        continue
    j = child["head"]
    par = by[child["parent"]]
    p = M[j].copy()
    d = (anchor(child["tail"]) - p) / np.linalg.norm(anchor(child["tail"]) - p) \
        + (p - anchor(par["head"])) / np.linalg.norm(p - anchor(par["head"]))
    d /= np.linalg.norm(d)
    start = p.copy()
    p = centre_in(p, d)
    if p is None:  # not inside the limb: leave it where SAM put it
        continue
    shift = p - start
    if np.linalg.norm(shift) > 0.04:
        p = start + shift / np.linalg.norm(shift) * 0.04
    M[j] = p
    print(f"{kp}: centred in the mesh, moved {np.linalg.norm(p - start) * 100:.1f} cm")

heads = {b["name"]: anchor(b["head"]) for b in bones}
tails = {b["name"]: anchor(b["tail"]) for b in bones}
kps = {}
for b in bones:
    if b.get("head_kp"):
        kps[b["head_kp"]] = (b["name"], "head", heads[b["name"]])
    if b.get("tail_kp"):
        kps[b["tail_kp"]] = (b["name"], "tail", tails[b["name"]])
parents = {b["name"]: b["parent"] for b in bones}


def prop_tip(hand):
    """Where a held prop ends, found from the mesh alone: of the surface joined to the hand that lies nearer the hand
    bone than any other bone, the point farthest from the wrist. None if nothing reaches past 1.5 hand lengths (an
    empty hand: the fingers end within one)."""
    seg = {n: (heads[n], tails[n]) for n in heads}

    def dist(a, b):
        ab = b - a
        t = np.clip(((co0 - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        return np.linalg.norm(co0 - a - np.outer(t, ab), axis=1)

    d = {n: dist(*s) for n, s in seg.items()}
    own = d[hand] <= np.min([d[n] for n in d if n != hand], axis=0)
    w, hl = heads[hand], np.linalg.norm(tails[hand] - heads[hand])
    reach = own & (np.linalg.norm(co0 - tails[hand], axis=1) < 0.5 * hl)
    e = E0[own[E0[:, 0]] & own[E0[:, 1]]]
    while True:
        grown = reach.copy()
        grown[e[reach[e[:, 1]], 0]] = True
        grown[e[reach[e[:, 0]], 1]] = True
        if (grown == reach).all():
            break
        reach = grown
    if not reach.any():
        return None
    r = np.linalg.norm(co0 - w, axis=1)
    i = np.flatnonzero(reach)[r[reach].argmax()]
    return co0[i] if r[i] > 1.5 * hl else None


for prop, spec in list(char.get("props", {}).items()):
    tip = spec.get("tip")
    if tip is None or tip == "auto":  # found from the mesh; a tip given in character.json overrides it
        tip = prop_tip(spec["parent"])
        if tip is None:
            print(f"{prop}: nothing held in {spec['parent']} reaches past the fingers: no prop")
            del char["props"][prop]
            continue
        print(f"{prop}: tip found at {np.round(tip, 3).tolist()}")
    spec["tip"] = list(map(float, tip))
    spec.setdefault("tip_kp", f"{prop}_tip")
    names.append(prop)
    heads[prop], tails[prop] = tails[spec["parent"]].copy(), np.array(spec["tip"], float)
    parents[prop] = spec["parent"]
    kps[spec["tip_kp"]] = (prop, "tail", tails[prop])
twists = {b["name"]: b["twist"] for b in bones if "twist" in b}

# Held props re-gripped: the generator put a dagger in a hammer grip, square to the forearm, and no wrist bends far
# enough to lay it along the arm. Whatever lies on its axis outside the fist (blade, guard, pommel) is turned about the
# middle of the fist until the prop sits grip_deg off the hand, and cut from the fingers it touched, which stay.
co = np.array([v.co[:] for v in body.data.vertices])
grip = {}
for prop, spec in char.get("props", {}).items():
    if spec.get("grip_deg") is None:
        continue
    hand, deg = spec["parent"], spec["grip_deg"]
    a, b = heads[prop], tails[prop]
    w, ht, el = heads[hand], tails[hand], heads[parents[hand]]
    L = np.linalg.norm(b - a)
    u = (b - a) / L
    t = (co - a) @ u
    d = np.linalg.norm(co - a - np.outer(t, u), axis=1)
    blade = (t > 0.4 * L) & (t < 0.8 * L) & (d < 0.04)
    c = co[blade].mean(0)  # the prop's own axis
    u = np.linalg.svd(co[blade] - c)[2][0]
    u *= np.sign(u @ (b - a))
    s = (co - c) @ u
    d = np.linalg.norm(co - c - np.outer(s, u), axis=1)
    hd = ht - w
    hl = np.linalg.norm(hd)
    th = np.clip(((co - w) @ hd) / hl ** 2, 0, 1)
    fe = el - w
    tf = np.clip(((co - w) @ fe) / (fe @ fe), 0, 1)
    dh = np.minimum(np.linalg.norm(w + np.outer(th, hd) - co, axis=1), np.linalg.norm(w + np.outer(tf, fe) - co, axis=1))
    near = d < 0.85 * hl
    fist = near & (dh < 0.85 * hl)  # the fingers round the grip: everything near the hand (wrist to knuckles) and forearm
    sp = s[fist].mean()
    sel = near & ~fist & (s < (b - c) @ u + 0.05) & (s > s[fist].min() - 2 * hl)
    # only what is joined to the blade or the fist: loose scraps that fall inside the cylinder (hair, cloth behind the
    # arm) would otherwise turn with the prop and stay fixed to the hand
    touch = np.zeros(len(co), bool)
    touch[E0[fist[E0[:, 1]], 0]] = True
    touch[E0[fist[E0[:, 0]], 1]] = True
    keep, e = sel & (blade | touch), E0[sel[E0[:, 0]] & sel[E0[:, 1]]]
    while True:
        grown = keep.copy()
        grown[e[keep[e[:, 1]], 0]] = True
        grown[e[keep[e[:, 0]], 1]] = True
        if (grown == keep).all():
            break
        keep = grown
    # a loose piece centred on the axis just past the fist is still the prop's (a pommel modelled as its own shell)
    lab = np.where(sel & ~keep, np.arange(len(co)), -1)
    el = e[(lab[e[:, 0]] >= 0) & (lab[e[:, 1]] >= 0)]
    while len(el):
        lo = np.minimum(lab[el[:, 0]], lab[el[:, 1]])
        new = lab.copy()
        np.minimum.at(new, el[:, 0], lo)
        np.minimum.at(new, el[:, 1], lo)
        if (new == lab).all():
            break
        lab = new
    for k in np.unique(lab[lab >= 0]):
        m = lab == k
        p = co[m].mean(0) - c
        if np.linalg.norm(p - (p @ u) * u) < 0.5 * hl and -hl < p @ u - s[fist].min() < 0:
            keep |= m
    if (sel & ~keep).any():
        print(f"{prop}: {int((sel & ~keep).sum())} vertices near its axis but not joined to it left alone")
    sel = keep
    n = np.cross(u, hd)
    n /= np.linalg.norm(n)
    ang = math.degrees(math.acos(np.clip(u @ hd / hl, -1, 1))) - deg
    rot = np.array(Matrix.Rotation(math.radians(ang), 3, Vector(n)))
    piv = c + sp * u
    co[sel] = (co[sel] - piv) @ rot.T + piv
    tails[prop] = (tails[prop] - piv) @ rot.T + piv
    kps[spec["tip_kp"]] = (prop, "tail", tails[prop])
    for v in np.nonzero(sel)[0]:
        body.data.vertices[v].co = co[v]
    grip[prop] = sel
    print(f"{prop}: re-gripped {ang:.0f} deg toward {hand}, {int(sel.sum())} vertices")
grip_mark = None
if grip:
    bm = bmesh.new()
    bm.from_mesh(body.data)
    mark = bm.verts.layers.int.new("grip")
    bm.verts.ensure_lookup_table()
    for k, sel in enumerate(grip.values(), 1):
        for v in np.nonzero(sel)[0]:
            bm.verts[v][mark] = k
    cut = [f for f in bm.faces if len({v[mark] for v in f.verts}) > 1]
    bmesh.ops.delete(bm, geom=cut, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    grip_mark = np.array([v[mark] for v in bm.verts])
    bm.to_mesh(body.data)
    bm.free()
    grip = {p: k for k, p in enumerate(grip, 1)}
    print(f"props cut from the fingers: {len(cut)} faces")

# the armature: local X across the body for bones in its plane, so flexion is about X
arm_data = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", arm_data)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
for n in names:
    e = arm_data.edit_bones.new(n)
    e.head, e.tail = Vector(heads[n]), Vector(tails[n])
    d = (e.tail - e.head).normalized()
    z = Vector((1, 0, 0)).cross(d)
    if z.length < 0.5:
        xl = Vector((0, 0, 1)).cross(d)
        z = xl.normalized().cross(d) if xl.length > 1e-4 else Vector((0, 0, 1))
    e.align_roll(z.normalized())
for n in names:
    if parents[n]:
        arm_data.edit_bones[n].parent = arm_data.edit_bones[parents[n]]
        arm_data.edit_bones[n].use_connect = False
bpy.ops.object.mode_set(mode="OBJECT")
use_mh = "makehuman" in tmpl and char.get("weights", "makehuman") == "makehuman"
for n in list(twists) + list(char.get("props", {})):  # the heat solve runs on the main bones; twist bones take
    arm_data.bones[n].use_deform = False  # their share afterwards, and a held prop is weighted to its hand
bpy.ops.object.select_all(action="DESELECT")
body.select_set(True)
arm.select_set(True)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.parent_set(type="ARMATURE_NAME" if use_mh else "ARMATURE_AUTO")
for n in twists:
    arm_data.bones[n].use_deform = True
    if n not in body.vertex_groups:
        body.vertex_groups.new(name=n)

nv = len(body.data.vertices)
co = np.array([v.co[:] for v in body.data.vertices])
gi = {g.name: g.index for g in body.vertex_groups}
W = np.zeros((nv, len(body.vertex_groups)))
for v in body.data.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight
segs = {n: (heads[n], tails[n]) for n in names}


def seg_dist(n):
    a, b = segs[n]
    t = np.clip(((co - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
    return np.linalg.norm(a + (b - a) * t[:, None] - co, axis=1)


E = np.array([e.vertices[:] for e in body.data.edges])
deg_ = np.bincount(E.ravel(), minlength=nv)[:, None].clip(1)
limbs = [side(c, s) for c in tmpl["limbs"] for s in ("L", "R")]
limb = {n for c in limbs for n in c}
for n in names:  # and what hangs off a limb (a prop)
    p = parents[n]
    while p and n not in limb:
        if p in limb:
            limb.add(n)
        p = parents[p]


def split_along(chain, blend=0.2):
    """The chain's share split again by where a vertex sits along it, blending a fraction of the shorter bone either
    side of each joint (the heat solve blends half a bone, and a bent wrist then pinches the forearm thin)."""
    ci = [gi[n] for n in chain]
    share = W[:, ci].sum(1)
    on = share > 1e-4
    ends = np.array([heads[chain[0]]] + [tails[n] for n in chain])
    for k in range(1, len(chain)):  # bones need not touch end to end: the chain runs through their heads
        ends[k] = heads[chain[k]]
    lens = np.linalg.norm(np.diff(ends, axis=0), axis=1)
    Lc = np.concatenate([[0], np.cumsum(lens)])
    best, s = np.full(nv, np.inf), np.zeros(nv)
    for k in range(len(chain)):
        a, b = ends[k], ends[k + 1]
        t = np.clip(((co - a) @ (b - a)) / (lens[k] ** 2), 0, 1)
        d = np.linalg.norm(a + (b - a) * t[:, None] - co, axis=1)
        near = d < best
        best[near], s[near] = d[near], Lc[k] + t[near] * lens[k]
    r = [np.ones(nv)]
    for j in range(1, len(chain)):
        h = blend * min(lens[j - 1], lens[j])
        x = np.clip((s - Lc[j] + h) / (2 * h), 0, 1)
        r.append(x * x * (3 - 2 * x))
    r.append(np.zeros(nv))
    for k, g in enumerate(ci):
        W[on, g] = share[on] * (r[k] - r[k + 1])[on]
    print(f"{chain[0]}..{chain[-1]}: weights split along it for {int(on.sum())} vertices")


def keep_own_side():
    """Each vertex belongs to the limb it is part of, found across the surface from vertices clearly on one side: the
    other side's weights below the limb's first bone go to the same bones of its own (two feet set close bleed)."""
    for chain in [c for c in limbs if c[0].endswith("_L")]:
        other = side(chain, "R")
        lower = [n for n in chain[1:] if n in gi]
        sides = [W[:, [gi[n] for n in c[1:]]].sum(1) for c in (chain, other)]
        label = np.full(nv, -1)
        for k in (0, 1):
            label[(sides[k] > 0.5) & (sides[1 - k] < 0.01)] = k
        while True:
            a, b = label[E[:, 0]], label[E[:, 1]]
            grow = np.concatenate([E[(a >= 0) & (b < 0)][:, [1, 0]], E[(b >= 0) & (a < 0)]])
            if not len(grow):
                break
            label[grow[:, 0]] = label[grow[:, 1]]
        fixed = 0
        for k, c, mine in ((0, other[1:], chain[1:]), (1, chain[1:], other[1:])):
            bad = (label == k) & (W[:, [gi[n] for n in c]].sum(1) > 0)
            for n, m in zip(c, mine):
                W[bad, gi[m]] += W[bad, gi[n]]
                W[bad, gi[n]] = 0
            fixed += int(bad.sum())
        print(f"{chain[0]}/{other[0]}: the other side's weights moved to their own on {fixed} vertices")


def rmin(a, b):
    """The least rotation taking unit vector a to unit vector b."""
    v = np.cross(a, b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + a @ b)


def frame(up, lateral):
    """Columns: lateral (made square to up), forward, up."""
    z = up / np.linalg.norm(up)
    x = lateral - (lateral @ z) * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def makehuman_weights():
    """MakeHuman's base body posed onto the skeleton: each vertex takes the weights of the nearest point of its body or
    skirt, the spine's share on the first torso bone. Returns the weights, each vertex's distance from that body, and
    the posed body (vertices, triangles) for the check renders."""
    spec = tmpl["makehuman"]
    mv, faces, J, raw = makehuman.load()
    sides = lambda k: ("L", "R") if k.endswith("_L") else ("L",)
    P = {side(k, s): J[b if s == "L" else b.replace(".L.", ".R.")] for k, b in spec["points"].items() for s in sides(k)}
    T = {k: kps[k][2] for k in P}
    s0 = T["head_top"][2] / P["head_top"][2]
    to = spec["torso"]
    up_p, up_t = P[to["top"]] - P[to["base"]], T[to["top"]] - T[to["base"]]
    seg = {}  # segment: (linear map, origin on MakeHuman, origin on ours, its turn)
    for name, l, org in (("torso_lo", to["hips"], to["base"]), ("torso_hi", to["shoulders"], to["top"])):
        r = side(l, "R")
        Fp, Ft = frame(up_p, P[l] - P[r]), frame(up_t, T[l] - T[r])
        S = np.diag([np.linalg.norm(T[l] - T[r]) / np.linalg.norm(P[l] - P[r]), s0,
                     np.linalg.norm(up_t) / np.linalg.norm(up_p)])
        seg[name] = (Ft @ S @ Fp.T, P[org], T[org], Ft @ Fp.T)
    todo = {side(b, s): [side(x, s) for x in v] for b, v in spec["segments"].items() for s in sides(b)}
    while todo:
        for b, (h, t, par) in list(todo.items()):
            if par in seg:
                dp, dt = P[t] - P[h], T[t] - T[h]
                up, ut = dp / np.linalg.norm(dp), dt / np.linalg.norm(dt)
                R = rmin(seg[par][3] @ up, ut) @ seg[par][3]
                S = s0 * np.eye(3) + (np.linalg.norm(dt) / np.linalg.norm(dp) - s0) * np.outer(up, up)
                seg[b] = (R @ S, P[h], T[h], R)
                del todo[b]
    cols = [b for b in seg if not b.startswith("torso_")] + ["torso"]
    Wm = makehuman.grouped_weights(raw, makehuman.rules_for(spec), len(mv), cols)
    move = lambda k, x: seg[k][2] + (x - seg[k][1]) @ seg[k][0].T
    f = np.clip(((mv - P[to["base"]]) @ up_p) / (up_p @ up_p), 0, 1)[:, None]
    f = f * f * (3 - 2 * f)
    X = np.zeros_like(mv)
    for c, b in enumerate(cols):
        Y = (1 - f) * move("torso_lo", mv) + f * move("torso_hi", mv) if b == "torso" else move(b, mv)
        X += Wm[:, c:c + 1] * Y

    def nearest(quads):
        """Each vertex's nearest point on these faces of the posed body: its weights and distance."""
        tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
        tree = BVHTree.FromPolygons([Vector(x) for x in X], tris.tolist())
        near = [tree.find_nearest(Vector(p)) for p in co]
        loc = np.array([n[0][:] for n in near])
        tri = tris[[n[2] for n in near]]
        a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]  # barycentric weights of the nearest point
        v0, v1, v2 = b - a, c - a, loc - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = (d00 * d11 - d01 * d01).clip(1e-12)
        bv = (d11 * d20 - d01 * d21) / den
        bw = (d00 * d21 - d01 * d20) / den
        bary = np.stack([1 - bv - bw, bv, bw], 1).clip(0, 1)
        bary /= bary.sum(1, keepdims=True)
        return (bary[:, :, None] * Wm[tri]).sum(1), np.array([n[3] for n in near]), tris

    # The body first. The skirt only for what hangs well off the body and nearer the skirt (a coat's panels): this
    # character's legs are thicker than MakeHuman's, and the inside of a trouser leg is nearer the skirt than the leg.
    Wc, dist, tris = nearest(faces["body"])
    if "helper-skirt" in faces:
        Ws, ds, st = nearest(faces["helper-skirt"])
        low = [cols.index(side(b, s)) for b in spec.get("no_skirt", []) for s in sides(b)]  # boots: not the hem
        k = (np.clip((dist - 0.03) / 0.04, 0, 1) * (ds < 0.5 * dist) * np.clip(1 - 2 * Wc[:, low].sum(1), 0, 1))[:, None]
        Wc = (1 - k) * Wc + k * Ws
        dist = np.where(k[:, 0] > 0, np.minimum(dist, ds), dist)
        tris = np.concatenate([tris, st])
        print(f"makehuman: {int((k > 0).sum())} vertices take the skirt's weights")
    out = np.zeros_like(W)
    for k, b in enumerate(cols):
        out[:, gi[to["bones"][0] if b == "torso" else b]] += Wc[:, k]
    print(f"makehuman: posed onto {len(seg) - 2} bones and the torso (scale {s0:.3f}); distance to its body: "
          f"median {np.median(dist) * 100:.1f} cm, 95% {np.percentile(dist, 95) * 100:.1f} cm")
    return out, dist, X, tris


if use_mh:
    W, dist, mh_fit, mh_tris = makehuman_weights()
    # clothing that hangs away from the body (coat tails, hair): weights spread from what sits on the body, 4 to 12 cm
    c = np.clip(1 - (dist - 0.04) / 0.08, 0, 1)[:, None]
    W0 = W.copy()
    live = np.nonzero(W.any(0))[0]
    for _ in range(200):
        acc = np.stack([np.bincount(E[:, 0], W[E[:, 1], g], nv) + np.bincount(E[:, 1], W[E[:, 0], g], nv)
                        for g in live], 1)
        W[:, live] = c * W0[:, live] + (1 - c) * acc / deg_
    print(f"weights spread onto {int((c < 1).sum())} vertices away from the body")
    # between the legs (the crotch) the nearest point found one leg or the other, vertex by vertex, and the seam tore
    # open as the legs parted; mixing both legs over a wide band webbed the inner thighs together instead. Right on the
    # seam the pelvis takes half and each leg a quarter, back to the vertex's own weights 6 rings of vertices away.
    pelvis = gi[tmpl["makehuman"]["torso"]["bones"][0]]
    for chain in [c for c in limbs if c[0].endswith("_L") and c[0] in tmpl["makehuman"].get("between", [])]:
        mine, theirs = [gi[n] for n in chain], [gi[n] for n in side(chain, "R")]
        share = W[:, mine + theirs].sum(1)
        left = W[:, mine].sum(1) / share.clip(1e-9)
        legs = share > 0.5
        seam = (np.abs(left[E[:, 0]] - left[E[:, 1]]) > 0.3) & legs[E[:, 0]] & legs[E[:, 1]]
        ring = np.full(nv, 99)
        ring[E[seam].ravel()] = 0
        for r in range(1, 7):
            near = np.zeros(nv, bool)
            near[E[ring[E[:, 1]] == r - 1, 0]] = True
            near[E[ring[E[:, 0]] == r - 1, 1]] = True
            ring[near & (ring > r) & legs] = r
        z = np.clip(1 - ring / 7, 0, 1)
        z = (z * z * (3 - 2 * z))[:, None]
        target = np.zeros_like(W)
        target[:, pelvis] = 0.5
        target[:, mine[0]] = target[:, theirs[0]] = 0.25
        W[:] = (1 - z) * W + z * target
        print(f"{chain[0]}/{side(chain[0], 'R')}: {int(seam.sum())} seam edges; pelvis and both legs blended on "
              f"{int((z[:, 0] > 0).sum())} vertices about it")
    # Below the crotch each leg is its own: the highest cut of the mesh below which the two knees are no longer joined
    # (the thighs of a scan in its rest pose often press together, and the seam above runs down between them); a leg's
    # vertices below it give up the pelvis and the other leg, fully 5 cm down.
    for chain in [c for c in limbs if c[0].endswith("_L") and c[0] in tmpl["makehuman"].get("between", [])]:
        other = side(chain, "R")
        knee = [np.argsort(np.linalg.norm(co - heads[c[1]], axis=1))[:20] for c in (chain, other)]

        def reach(seed, h):
            on = np.zeros(nv, bool)
            on[seed] = True
            below = co[:, 2] < h
            while True:
                grow = below[E[:, 0]] & below[E[:, 1]] & (on[E[:, 0]] != on[E[:, 1]])
                if not grow.any():
                    return on
                on[E[grow].ravel()] = True

        lo = crotch_z if crotch_z is not None else parting(heads[chain[1]], heads[other[1]], heads[chain[1]][2],
                                                           heads[chain[0]][2] + 0.05)
        drop = gi[tmpl["makehuman"]["torso"]["bones"][0]]
        for c, o in ((chain, other), (other, chain)):
            leg = reach(knee[0] if c is chain else knee[1], lo)
            keep = np.clip((co[:, 2] - (lo - 0.05)) / 0.05, 0, 1)
            for g in [drop] + [gi[n] for n in o]:
                give = np.where(leg, W[:, g] * (1 - keep), 0)
                W[:, g] -= give
                W[:, gi[c[0]]] += give
        print(f"{chain[0]}/{other[0]}: the legs part {lo:.3f} m up; below it each keeps to itself")
    # nothing below a bone's parent follows it (a coat's collar is nearer MakeHuman's chin than its chest): its weight
    # fades out down the length of its parent (the chin hangs below the head's joint) and goes to the parent
    for n in tmpl["makehuman"].get("not_below", []):
        par = parents[n]
        a, b = heads[par], heads[n]
        t = ((co - b) @ (b - a)) / ((b - a) @ (b - a))
        keep = np.clip(1 + t, 0, 1)
        give = W[:, gi[n]] * (1 - keep)
        W[:, gi[n]] -= give
        W[:, gi[par]] += give
        print(f"{n}: weight below its joint given to {par} on {int((give > 0.01).sum())} vertices")
    # a vertex clearly nearer one limb's bones than the other side's gives up that other limb's weights to the same
    # bones of its own (thick legs close together); halfway between (a coat panel) it keeps both
    for chain in [c for c in limbs if c[0].endswith("_L")]:
        other = side(chain, "R")
        d = [np.min([seg_dist(n) for n in c], 0) for c in (chain, other)]
        moved = np.zeros(nv, bool)
        for k, c, mine in ((0, other, chain), (1, chain, other)):
            m = np.clip((d[1 - k] / np.maximum(d[k], 1e-6) - 1.15) / 0.3, 0, 1)
            for n, own in zip(c, mine):
                give = W[:, gi[n]] * m
                moved |= give > 0.01
                W[:, gi[own]] += give
                W[:, gi[n]] -= give
        print(f"{chain[0]}/{other[0]}: the other side's weights moved to their own on {int(moved.sum())} vertices")
    keep_own_side()
    split_along(tmpl["makehuman"]["torso"]["bones"], blend=0.5)
else:
    main = [n for n in names if n not in twists and n in gi]
    empty = W.sum(1) < 1e-3  # vertices the heat solve missed: the nearest bone
    if empty.any():
        d = np.stack([seg_dist(n)[empty] for n in main], 1)
        W[np.nonzero(empty)[0], [gi[main[j]] for j in d.argmin(1)]] = 1.0
    print("vertices without weights fixed:", int(empty.sum()))
    for _ in range(3):
        acc = np.zeros_like(W)
        np.add.at(acc, E[:, 0], W[E[:, 1]])
        np.add.at(acc, E[:, 1], W[E[:, 0]])
        W = 0.5 * W + 0.5 * acc / deg_

    # a limb's weight fades with distance from its bone, 5 to 10% of the character's size, to the vertex's other bones
    size = np.ptp(co, axis=0).max()
    before = W.copy()
    for n in limb & set(main):
        W[:, gi[n]] *= np.clip((0.10 * size - seg_dist(n)) / (0.05 * size), 0, 1)
    empty = W.sum(1) < 1e-3
    if empty.any():
        torso = [n for n in main if n not in limb]
        d = np.stack([seg_dist(n)[empty] for n in torso], 1)
        W[np.nonzero(empty)[0], [gi[torso[j]] for j in d.argmin(1)]] = 1.0
    print(f"limb weight pulled back from {int(((before - W).clip(0).sum(1) > 0.05).sum())} vertices far from their limb")

    keep_own_side()

    for chain in limbs:
        split_along(chain)

# Tethered limbs (the template's "tethered", the arms): each bone's weight is kept to what lies near its limb along
# the surface. Weights copied from the nearest body surface reach loose clothing hanging beside the limb (a coat's flank
# by a hanging elbow), close in space but far along the mesh (up through the armpit and down again), and the limb then
# drags it into long bands. Each bone's own radius is measured on this mesh (the median distance of the vertices it
# surely moves); from the vertices within 1.25 radii, its weight holds for 2 radii along the surface and is gone by 4.
EL = np.linalg.norm(co[E[:, 0]] - co[E[:, 1]], axis=1)


def along_surface(seed, cap):
    """Distance along the mesh's edges from the seed vertices, up to cap (inf beyond)."""
    g = np.where(seed, 0.0, np.inf)
    for _ in range(1000):
        new = g.copy()
        np.minimum.at(new, E[:, 0], g[E[:, 1]] + EL)
        np.minimum.at(new, E[:, 1], g[E[:, 0]] + EL)
        new[new > cap] = np.inf
        if np.array_equal(new, g):
            break
        g = new
    return g


tethered = {side(n, s) for n in tmpl.get("tethered", []) for s in ("L", "R")} & set(gi)
cut = 0
for n in sorted(tethered):
    w, d = W[:, gi[n]], seg_dist(n)
    if (w > 0.9).sum() < 20:
        continue
    r = float(np.median(d[w > 0.9]))
    g = along_surface((d < 1.25 * r) & (w > 0.9), 4 * r)  # both walls of a double-walled surface: they only meet at
    keep = np.clip((4 * r - g) / (2 * r), 0, 1)               # its edges
    cut += int(((w > 0.05) & (keep < 1)).sum())
    W[:, gi[n]] = w * keep
lost = W.sum(1) < 1e-3
if lost.any():
    torso = [n for n in names if n in gi and n not in limb and n not in twists and n not in char.get("props", {})]
    d = np.stack([seg_dist(n)[lost] for n in torso], 1)
    W[np.nonzero(lost)[0], [gi[torso[j]] for j in d.argmin(1)]] = 1.0
if tethered:
    print(f"tethered limbs: weight kept to what lies near them along the surface, cut on {cut} vertices")

# twist bones: the part of their limb bone nearest their own end of it, blended over the middle fifth
for n, tw in twists.items():
    a, b = segs[tw["of"]]
    t = np.clip(((co - a) @ (b - a)) / ((b - a) @ (b - a)), 0, 1)
    if tw["from"] == "tail":
        t = 1 - t
    x = np.clip((t - (tw["section"] - 0.2)) / 0.4, 0, 1)
    f = 1 - x * x * (3 - 2 * x)
    W[:, gi[n]] = W[:, gi[tw["of"]]] * f
    W[:, gi[tw["of"]]] *= 1 - f
    print(f"{n}: {int((W[:, gi[n]] > 0.01).sum())} vertices of {tw['of']}")

keep = np.argsort(-W, axis=1)[:, :4]
mask = np.zeros_like(W, bool)
np.put_along_axis(mask, keep, True, axis=1)
W = np.where(mask, W, 0)
W /= W.sum(1, keepdims=True).clip(1e-9)

# loose pieces (a buckle the generator left floating) move with the closest vertex of the main surface
parent_ = list(range(nv))


def find(i):
    while parent_[i] != i:
        parent_[i] = parent_[parent_[i]]
        i = parent_[i]
    return i


for a, b in E:
    ra, rb = find(int(a)), find(int(b))
    if ra != rb:
        parent_[ra] = rb
piece = np.array([find(i) for i in range(nv)])
ids, counts = np.unique(piece, return_counts=True)
body_v = np.nonzero(piece == ids[counts.argmax()])[0]
loose = 0
for pid, n in zip(ids, counts):
    if n < 0.02 * nv:
        idx = np.nonzero(piece == pid)[0]
        sub = idx[:: max(1, len(idx) // 64)]
        d = ((co[sub, None] - co[None, body_v]) ** 2).sum(-1)
        W[idx] = W[body_v[d.min(0).argmin()]]
        loose += 1
print("loose pieces moving with the nearest surface:", loose)
for prop, k in grip.items():  # a re-gripped prop is fixed in its hand: its own bone only carries the tip's reading
    hand = gi[parents[prop]]
    W[grip_mark == k] = 0
    W[grip_mark == k, hand] = 1
    print(f"{prop}: fixed in {parents[prop]}, {int((grip_mark == k).sum())} vertices")
for g, vg in enumerate(body.vertex_groups):
    vg.remove(list(range(nv)))
    for i in np.nonzero(W[:, g] > 1e-4)[0]:
        vg.add([int(i)], float(W[i, g]), "REPLACE")
print("weights written; max bones per vertex:", int((W > 1e-4).sum(1).max()))

# rig_info.json: what the retarget, the renderer and the studio read (anymove's layout, so its renderer works)
lay = dict(char.get("layout", {}))
lay.setdefault("H", 720)
lay["Z_CENTER"] = (lay["GROUND_V"] - lay["H"] / 2) * lay["M_PER_PX"]
lay.update({"Y_CENTER": 0.0, "X_CENTER": 0.0})
info = {"bones": [], "keypoints": [], "layout": lay, "template": pathlib.Path(tmpl_path).stem}
for n in names:
    bb = arm_data.bones[n]
    hk = next((k for k, v in kps.items() if v[0] == n and v[1] == "head"), f"{n}.head")
    tk = next((k for k, v in kps.items() if v[0] == n and v[1] == "tail"), f"{n}.tail")
    info["bones"].append({"name": n, "parent": names.index(parents[n]) if parents[n] else -1,
                          "matrix_local": [list(r) for r in bb.matrix_local], "length": bb.length,
                          "head_joint": hk, "tail_joint": tk, "limits_deg": [[-180, 180]] * 3})
for k, (n, end, p) in kps.items():
    info["keypoints"].append({"name": k, "bone": names.index(n), "end": end, "rest": [round(float(x), 5) for x in p]})
json.dump(info, open(out / "rig_info.json", "w"), indent=1)

rt = tmpl["retarget"]
both = lambda d: {side(k, s): side(v, s) for k, v in d.items() for s in (("L", "R") if k.endswith("_L") else ("L",))}
rmap = {"map": both(rt["map"]), "aim": both(rt["aim"]),
        "props": {p: {"base": next(k for k, v in kps.items() if v[0] == s["parent"] and v[1] == "tail"),  # the hand's tip
                      "tip": s["tip_kp"], "along": s.get("along", 0)} for p, s in char.get("props", {}).items()},
        **{k: v for k, v in rt.items() if k not in ("map", "aim")}}
json.dump(rmap, open(out / "retarget_map.json", "w"), indent=1)

bpy.ops.wm.save_as_mainfile(filepath=str(out / "rigged.blend"))
print("saved", out / "rigged.blend")

# a check: the joints over an x-ray of the mesh, front and side
scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.show_xray = True
scene.display.shading.xray_alpha = 0.25
scene.display.shading.color_type = "OBJECT"
scene.render.resolution_x, scene.render.resolution_y = 600, 900
for k, (n, end, p) in kps.items():
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.015, location=Vector(p))
    bpy.context.object.color = (1, 0.2, 0.1, 1)
for n in names:
    a, b = Vector(heads[n]), Vector(tails[n])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=(b - a).length, location=(a + b) / 2)
    bpy.context.object.rotation_mode = "QUATERNION"
    bpy.context.object.rotation_quaternion = (b - a).to_track_quat("Z", "Y")
    bpy.context.object.color = (0.1, 0.6, 1, 1)
body.color = (0.8, 0.8, 0.8, 1)
cam = bpy.data.objects.new("check_cam", bpy.data.cameras.new("check_cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = 2.0
for view, loc, rot in (("front", (0, -6, 0.9), (math.pi / 2, 0, 0)), ("side", (6, 0, 0.9), (math.pi / 2, 0, math.pi / 2))):
    cam.location, cam.rotation_euler = loc, rot
    scene.render.filepath = str(out / f"joints_{view}.png")
    bpy.ops.render.render(write_still=True)
print("check renders:", out / "joints_front.png", out / "joints_side.png")
if use_mh:  # and MakeHuman's body as posed for the weights, over the character
    for o in list(scene.objects):
        if o.type == "MESH" and o is not body:
            o.hide_render = True
    me = bpy.data.meshes.new("makehuman_fit")
    me.from_pydata([tuple(x) for x in mh_fit], [], mh_tris.tolist())
    fit = bpy.data.objects.new("makehuman_fit", me)
    scene.collection.objects.link(fit)
    fit.color = (1, 0.3, 0.2, 1)
    for view, loc, rot in (("front", (0, -6, 0.9), (math.pi / 2, 0, 0)), ("side", (6, 0, 0.9), (math.pi / 2, 0, math.pi / 2))):
        cam.location, cam.rotation_euler = loc, rot
        scene.render.filepath = str(out / f"makehuman_{view}.png")
        bpy.ops.render.render(write_still=True)
    print("check renders:", out / "makehuman_front.png", out / "makehuman_side.png")
