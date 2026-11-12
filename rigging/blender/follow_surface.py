"""Loose pieces of the mesh lying on its surface: a piece that only doubles the surface is removed, one that sits on it
moves with it. A loose piece is a small connected part of the body mesh not joined to the main one.

blender -b rigged.blend -P rigging/blender/follow_surface.py -- [--max-share 0.05] [--near 0.03] [--double 0.004]

TRELLIS builds detail it can't resolve as loose pieces on the surface: the rogue's narrow coat slit became 22 strips,
2 cm tall, lying within a millimetre of the flap. Weighted on their own, each took the nearest leg (half of them the
other leg from the flap under them, which the cloth bones carry) and in any stride they fanned out of the flap like a
comb; given the flap's weights they still flickered through it, as two surfaces a millimetre apart can't move as one.
- a piece whose vertices lie, at the median, within --double metres of the main mesh doubles its surface: it adds
  nothing to the look (the rogue's back looks the same without them) and is removed;
- a piece within --near metres sits on the surface (a strap, a trim): each vertex takes the weights of the nearest
  point of the main mesh, so it moves with what it lies on;
- a piece further away (or more than --max-share of the mesh's vertices) keeps its own weights.
Run after the cloth (add_cloth.py), so pieces follow the cloth's weights too. Rewrites the .blend in place.
"""
import sys

import bmesh
import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
opt = lambda k, d: float(args[args.index(k) + 1]) if k in args else d
max_share, near, double = opt("--max-share", 0.05), opt("--near", 0.03), opt("--double", 0.004)

body = bpy.data.objects["body"]
me = body.data
nv, ng = len(me.vertices), len(body.vertex_groups)
co = np.array([v.co[:] for v in me.vertices])
W = np.zeros((nv, ng))
for v in me.vertices:
    for g in v.groups:
        W[v.index, g.group] = g.weight

# connected pieces
E = np.array([e.vertices[:] for e in me.edges])
lab = np.arange(nv)
while True:
    lo = np.minimum(lab[E[:, 0]], lab[E[:, 1]])
    new = lab.copy()
    np.minimum.at(new, E[:, 0], lo)
    np.minimum.at(new, E[:, 1], lo)
    new = new[new]  # jump along chains: converges in a few rounds instead of one per vertex of the longest path
    if np.array_equal(new, lab):
        break
    lab = new
ids, counts = np.unique(lab, return_counts=True)
main = ids[np.argmax(counts)]
in_main = lab == main

bm = bmesh.new()
bm.from_mesh(me)
bm.verts.ensure_lookup_table()
face_verts = [[v.index for v in f.verts] for f in bm.faces if in_main[f.verts[0].index]]
tree = BVHTree.FromPolygons([v.co for v in bm.verts], face_verts)

dropped, moved, kept = [], [], []
follow = np.zeros(nv, bool)
for k, n in zip(ids, counts):
    if k == main or n > max_share * nv:
        continue
    piece = np.flatnonzero(lab == k)
    hits = [tree.find_nearest(bm.verts[i].co) for i in piece]
    gap = np.median([h[3] for h in hits])
    if gap <= double:
        dropped.append(piece)
    elif gap <= near:
        for i, (loc, _, fi, _) in zip(piece, hits):
            vs = face_verts[fi]
            w = 1 / np.maximum(np.linalg.norm(co[vs] - np.array(loc), axis=1), 1e-5)
            W[i] = (w[:, None] * W[vs]).sum(0) / w.sum()
        follow[piece] = True
        moved.append(int(n))
    else:
        kept.append(int(n))

for i in np.flatnonzero(follow):
    w = W[i].copy()
    w[np.argsort(w)[::-1][4:]] = 0  # at most four bones a vertex, as the game's skinning takes
    w /= max(w.sum(), 1e-9)
    for g in body.vertex_groups:
        if w[g.index] > 1e-4:
            g.add([int(i)], float(w[g.index]), "REPLACE")
        else:
            g.remove([int(i)])
if dropped:  # last: removing vertices renumbers them
    gone = np.concatenate(dropped)
    bm.free()
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[bm.verts[int(i)] for i in gone], context="VERTS")
    bm.to_mesh(me)
bm.free()
if dropped or moved:
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_mainfile()
print(f"follow surface: {len(dropped)} loose pieces doubling the surface removed ({sum(map(len, dropped))} vertices), "
      f"{len(moved)} lying on it move with it ({sum(moved)} vertices)"
      + (f", {len(kept)} apart from it keep their own weights ({sum(kept)} vertices)" if kept else ""))
