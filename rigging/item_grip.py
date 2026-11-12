"""Where an item is held: its grip, estimated from its mesh, written to the item's item.json.

    python -m rigging.item_grip ITEM_DIR [MODEL.glb] [--sockets hand,hip] [--length-m 0.6 [--length-from "..."]]

Coordinates are the GLB's own (glTF: y up, the front of what TRELLIS made facing +z), so a game reads them as they are.
The grip is a frame: its origin goes on a rig's socket (rigging/blender/add_sockets.py), its +Y towards the item's tip,
its +Z out of its face.

From the item's width along its length (the long axis of its vertices):
- the widest place in from both ends (more than 12% of the length from each) is a guard: the grip is on its shorter
  side, its middle 45% of the way from the guard to that end (a pommel is left beyond it), +Y towards the longer side
  (a dagger, a sword);
- the widest place at an end is a head: the grip is 20% of the length in from the other end, +Y towards the head (an
  axe, a hammer, a mace);
+Z: the item's thinnest direction across its length (a blade's flat), signed to the front of the item's picture.
Anything else (a bow, a shield) needs its grip set by hand: edit item.json's "grip" (origin, x, y, z).

TRELLIS makes every model about one unit long: "length_m" is the item's real length (--length-m, with where that
came from in "length_from"; kept from an earlier item.json), "length_units" its length in the model, so the model
is scaled by length_m / length_units.
"""
import argparse, json
from pathlib import Path

import numpy as np
import trimesh


def vertices(model):
    """Every vertex of a GLB, in its own (glTF) coordinates."""
    scene = trimesh.load(model, force="scene")
    return np.concatenate([g.apply_transform(t).vertices for g, t in
                           ((scene.geometry[n].copy(), scene.graph[n][0]) for n in scene.graph.nodes_geometry)])


def estimate(v):
    c = v.mean(0)
    axes = np.linalg.svd(v - c, full_matrices=False)[2]
    u = axes[0]
    s = (v - c) @ u
    lo, hi = s.min(), s.max()
    L = hi - lo
    bins = np.linspace(lo, hi, 41)
    across = v - c - np.outer(s, u)
    width = np.array([np.linalg.norm(across[(s >= a) & (s < b)], axis=1).max(initial=0) for a, b in zip(bins, bins[1:])])
    mid = (bins[:-1] + bins[1:]) / 2
    widest = mid[width.argmax()]
    if lo + 0.12 * L < widest < hi - 0.12 * L:
        far, near = (hi, lo) if widest - lo < hi - widest else (lo, hi)  # near: the shorter side's end
        origin_s = widest + 0.45 * (near - widest)
        y = u if far > widest else -u
        how = "guard"
    else:
        head, other = (hi, lo) if abs(widest - hi) < abs(widest - lo) else (lo, hi)
        origin_s = other + 0.2 * (head - other)
        y = u if head > other else -u
        how = "head"
    origin = c + origin_s * u
    z = axes[2] - (axes[2] @ y) * y
    z /= np.linalg.norm(z)
    if z[2] < 0:  # glTF +z: the front of the item's picture
        z = -z
    x = np.cross(y, z)
    return {"origin": origin, "x": x, "y": y, "z": z, "how": how, "length": float(L)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--sockets", default="hand")
    ap.add_argument("--length-m", type=float)
    ap.add_argument("--length-from")
    a = ap.parse_args()
    d = Path(a.item).resolve()
    meta = json.loads((d / "item.json").read_text()) if (d / "item.json").exists() else {}
    model = Path(a.model).resolve() if a.model else (d / meta["model"] if meta.get("model") else None)
    if model is None:
        models = sorted(d.glob("*/*_textured.glb"), key=lambda p: p.stat().st_mtime)
        if not models:
            raise SystemExit(f"no model in {d}: make one (TRELLIS) or name it")
        model = models[-1]
    g = estimate(vertices(model))
    r = lambda x: [round(float(t), 5) for t in x]
    meta.update(model=str(model.relative_to(d)), sockets=a.sockets.split(","),
                grip={"origin": r(g["origin"]), "x": r(g["x"]), "y": r(g["y"]), "z": r(g["z"]), "estimated": g["how"]},
                length_units=round(g["length"], 4))
    if a.length_m:
        meta.update(length_m=a.length_m, length_from=a.length_from or "given")
    (d / "item.json").write_text(json.dumps(meta, indent=1))
    print(f"{d.name}: grip by its {g['how']}, origin {r(g['origin'])}, +Y {np.round(g['y'], 2).tolist()}, "
          f"+Z {np.round(g['z'], 2).tolist()}, {g['length']:.3f} long in the model's units"
          + (f" ({meta['length_m']} m)" if meta.get("length_m") else ", real length not given (--length-m)")
          + f" -> {d / 'item.json'}")


if __name__ == "__main__":
    main()
