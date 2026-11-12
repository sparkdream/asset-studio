"""Loose cloth on spring bones: coat panels, skirts, tabards, a cape's hem, whatever hangs off the body below the hips.

blender -b -P rigging/blender/add_cloth.py -- rigged.blend [character.json]

Copied weights tie such cloth to the legs, which then drag it into a stretched sheet (the rig check's "skin weights
reach too far"). Here it is found and given its own chains of spring bones, simulated by apply_motion.py --springs as the
hair is, so it swings with the movement instead:
- loose cloth: vertices below the hip joints that a leg moves (weight > 0.2) but that lie further from every leg bone
  than 1.6 times that leg's own thickness (the median distance of the vertices it surely moves); each connected piece of
  at least `min_vertices` is a panel. Nothing about any one character: the legs are the template limbs hanging from the
  root bone.
- bones: a chain of `segments` down each panel, from its top to its hem, parented to the root bone.
- weights: a panel's leg weight goes to the root bone (the hips carry it, not the thighs); the chain takes over from
  `free` of the way down, rising to `tip` at the hem; the weights are blended over `rings` rings round the panel
  (16, about 11 cm on a TRELLIS mesh: 2 only moved the stretch to the seam between panel and coat).
character.json "cloth": {...} changes any of CLOTH's settings, or "cloth": false turns it off. Rewrites rigged.blend in
place and adds the bones to springs.json beside it (hair springs, if any, are kept).
"""
import json, pathlib, sys

import bpy
import numpy as np
from mathutils import Vector

CLOTH = {"min_vertices": 300, "thickness": 1.6, "segments": 3, "free": 0.2, "tip": 0.9, "rings": 16, "smooth": 30,
         "freq": [2.2, 1.8, 1.5], "damping": 0.4, "max_deg": [12, 20, 28], "gravity": 0.8}

args = sys.argv[sys.argv.index("--") + 1:]
blend = pathlib.Path(args[0])
char = json.load(open(args[1])) if len(args) > 1 else {}
if char.get("cloth") is False:
    sys.exit(print("cloth: off in character.json"))
CLOTH.update(char.get("cloth") or {})
bpy.ops.wm.open_mainfile(filepath=str(blend))
rig, body = bpy.data.objects["rig"], bpy.data.objects["body"]
if any(b.name.startswith("cloth_") for b in rig.data.bones):
    sys.exit("this rig already has its cloth bones: rebuild it (rigging/rig.sh) first")
info = json.load(open(blend.parent / "rig_info.json"))
tmpl = json.load(open(pathlib.Path(__file__).resolve().parents[1] / "templates" / f"{info.get('template', 'humanoid')}.json"))

me = body.data
nv = len(me.vertices)
co = np.array([body.matrix_world @ v.co for v in me.vertices])
E = np.array([e.vertices[:] for e in me.edges])
gn = [g.name for g in body.vertex_groups]
W = np.zeros((nv, len(gn)))
for v in me.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight
bones = rig.data.bones
root = next(b for b in bones if b.parent is None).name
R = rig.matrix_world


def seg(n):
    b = bones[n]
    return np.array(R @ b.head_local), np.array(R @ b.tail_local)


def seg_dist(n):
    a, b = seg(n)
    t = np.clip(((co - a) @ (b - a)) / max((b - a) @ (b - a), 1e-12), 0, 1)
    return np.linalg.norm(a + (b - a) * t[:, None] - co, axis=1)


# the legs: template limbs hanging from the root bone, with every bone below them (twist bones, toes)
side = lambda n, s: n[:-2] + "_" + s if n.endswith(("_L", "_R")) else n
legs = []
for chain in tmpl["limbs"]:
    for s in ("L", "R"):
        first = side(chain[0], s)
        if first in bones and bones[first].parent and bones[first].parent.name == root:
            legs.append(first)
leg_bones = set()
for first in legs:
    todo = [bones[first]]
    while todo:
        b = todo.pop()
        leg_bones.add(b.name)
        todo += list(b.children)
leg_bones &= set(gn)
if not leg_bones:
    sys.exit(print("cloth: no legs hanging from the root bone in this rig"))
gi = {n: i for i, n in enumerate(gn)}
leg_w = W[:, [gi[n] for n in leg_bones]].sum(1)
dist = {n: seg_dist(n) for n in leg_bones}
near = np.min(np.stack(list(dist.values())), 0)
reach = np.zeros(nv, bool)
for n in leg_bones:
    core = W[:, gi[n]] > 0.9
    if core.sum() < 20:
        continue
    reach |= dist[n] <= CLOTH["thickness"] * np.median(dist[n][core])
hips = min(seg(f)[0][2] for f in legs)
seed = (leg_w > 0.2) & ~reach & (co[:, 2] < hips)
loose = ~reach & (co[:, 2] < hips)  # what a panel may grow through: below the hips, off the legs

# panels: connected pieces of loose cloth holding a seed
lab = np.where(loose, np.arange(nv), -1)
el = E[loose[E[:, 0]] & loose[E[:, 1]]]
while True:
    lo = np.minimum(lab[el[:, 0]], lab[el[:, 1]])
    new = lab.copy()
    np.minimum.at(new, el[:, 0], lo)
    np.minimum.at(new, el[:, 1], lo)
    if np.array_equal(new, lab):
        break
    lab = new
panels = [np.flatnonzero(lab == k) for k in np.unique(lab[seed])]
panels = [p for p in panels if len(p) >= CLOTH["min_vertices"]]
# cloth hangs from the body: a panel must border vertices the torso carries (leg weight under a half). A loose boot top
# or trouser cuff borders only its leg, and must keep moving with it.
held = []
for p in panels:
    m = np.zeros(nv, bool)
    m[p] = True
    rim = np.unique(np.concatenate([E[m[E[:, 0]] & ~m[E[:, 1]], 1], E[m[E[:, 1]] & ~m[E[:, 0]], 0]]))
    if (leg_w[rim] < 0.5).sum() >= 10:
        held.append(p)
    else:
        print(f"cloth: a loose piece of {len(p)} vertices hangs from a leg, not the body: left on the leg")
panels = held
if not panels:
    sys.exit(print("cloth: no loose cloth found"))

# a chain down each panel: the middle of the panel at even heights from its top to its hem
chains = []
for p in panels:
    P = co[p]
    top, hem = P[:, 2].max(), np.percentile(P[:, 2], 2)
    pts = []
    for z in np.linspace(top, hem, CLOTH["segments"] + 1):
        sl = np.abs(P[:, 2] - z) < 0.03
        pts.append(np.append(P[sl, :2].mean(0) if sl.sum() > 10 else P[np.argsort(np.abs(P[:, 2] - z))[:20], :2].mean(0), z))
    chains.append(np.array(pts))
print("cloth panels:", len(panels), "| vertices:", [len(p) for p in panels],
      "| lengths (m):", [round(float(c[0, 2] - c[-1, 2]), 2) for c in chains])

bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="EDIT")
names_all = []
for k, pts in enumerate(chains):
    names = []
    for j in range(len(pts) - 1):
        n = f"cloth_{k}_{j}"
        b = rig.data.edit_bones.new(n)
        b.head, b.tail = Vector(pts[j].tolist()), Vector(pts[j + 1].tolist())
        zv = Vector((1, 0, 0)).cross((b.tail - b.head).normalized())
        b.align_roll(zv.normalized() if zv.length > 1e-4 else Vector((0, 0, 1)))
        b.parent = rig.data.edit_bones[names[-1] if names else root]
        names.append(n)
    names_all.append(names)
bpy.ops.object.mode_set(mode="OBJECT")

# weights: a panel's leg share to the root bone; the chain takes over toward the hem; the edge smoothed
cols = dict(gi)
for names in names_all:
    for n in names:
        cols[n] = len(cols)
Wn = np.hstack([W, np.zeros((nv, len(cols) - len(gn)))])
smooth01 = lambda x: (lambda c: c * c * (3 - 2 * c))(np.clip(x, 0, 1))
in_panel = np.zeros(nv, bool)
for p, pts, names in zip(panels, chains, names_all):
    in_panel[p] = True
    legs_share = Wn[p][:, [cols[n] for n in leg_bones]].sum(1)
    Wn[np.ix_(p, [cols[n] for n in leg_bones])] = 0
    Wn[p, cols[root]] += legs_share
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    s = np.clip((pts[0, 2] - co[p, 2]) / max(pts[0, 2] - pts[-1, 2], 1e-6), 0, 1) * L[-1]  # how far down the panel
    f = CLOTH["tip"] * smooth01((s / L[-1] - CLOTH["free"]) / (1 - CLOTH["free"]))
    Wn[p] *= (1 - f)[:, None]
    for j, n in enumerate(names):
        h = 0.25 * (L[j + 1] - L[j])
        a = np.ones(len(p)) if j == 0 else smooth01((s - L[j] + h) / (2 * h))
        b = np.zeros(len(p)) if j == len(names) - 1 else smooth01((s - L[j + 1] + h) / (2 * h))
        Wn[p, cols[n]] += f * (a - b)
deg = np.bincount(E.ravel(), minlength=nv).clip(1)[:, None]
edge = in_panel.copy()  # the panel and `rings` rings of body round it, blended over `smooth` passes
for _ in range(CLOTH["rings"]):
    edge[E[edge[E[:, 0]], 1]] = True
    edge[E[edge[E[:, 1]], 0]] = True
for _ in range(CLOTH["smooth"]):
    acc = np.zeros_like(Wn)
    np.add.at(acc, E[:, 0], Wn[E[:, 1]])
    np.add.at(acc, E[:, 1], Wn[E[:, 0]])
    Wn[edge] = 0.5 * Wn[edge] + 0.5 * (acc / deg)[edge]
Wn /= Wn.sum(1, keepdims=True).clip(1e-9)

groups = {g.name: g for g in body.vertex_groups}
for names in names_all:
    for n in names:
        groups[n] = body.vertex_groups.new(name=n)
inv = {i: n for n, i in cols.items()}
changed = np.flatnonzero(edge)
for i in changed:
    for g in list(me.vertices[i].groups):
        body.vertex_groups[g.group].remove([int(i)])
    for j in np.flatnonzero(Wn[i] > 1e-4):
        groups[inv[j]].add([int(i)], float(Wn[i, j]), "REPLACE")
print(f"cloth weights: {int(in_panel.sum())} panel vertices, {len(changed)} with weights changed (the edge smoothed)")

sp = blend.parent / "springs.json"
springs = json.load(open(sp)) if sp.exists() else {}
springs = {n: v for n, v in springs.items() if not n.startswith("cloth_")}
for names in names_all:
    for j, n in enumerate(names):
        pick = lambda v: v[min(j, len(v) - 1)] if isinstance(v, list) else v
        springs[n] = {"parent": names[j - 1] if j else root, "freq": pick(CLOTH["freq"]), "damping": pick(CLOTH["damping"]),
                      "max_deg": pick(CLOTH["max_deg"]), "gravity": pick(CLOTH["gravity"])}
json.dump(springs, open(sp, "w"), indent=1)
bpy.ops.wm.save_as_mainfile(filepath=str(blend))
print("saved", blend, "and", sp)
