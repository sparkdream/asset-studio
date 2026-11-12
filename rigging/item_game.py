"""An item ready for the game: its grip (rigging/item_grip.py) and a light copy, aligned to a socket.

    PYTHONPATH=external/trellis2/o-voxel python -m rigging.item_game ITEM_DIR MODEL.glb [--faces 5000]
        [--texture 1024] [--length-m 0.6 [--length-from "..."]] [--if-new]

MODEL.glb is one of the item's TRELLIS models; its run's sample_xpu.pt (the voxels with their colours) is baked again
into a light copy, ITEM_DIR/game/<name>.glb, as postprocess_xpu.py does: --faces, a --texture-sized texture (painted from
the run's views again when MODEL.glb is a reprojected one, reference/reproject.py). The copy is
moved so its grip is its origin and the grip's axes are its own (+Y to the tip, +Z out of its face), and scaled to its
real length in metres: under a rig's socket node (rigging/blender/add_sockets.py; the glTF exporter keeps each bone's
own axes on its node) it needs no transform of its own. In three.js: socket.add(item).

The grip is estimated from MODEL.glb's shape, unless item.json's grip says "estimated": "by hand" for that same model.
Without a real length (--length-m, or kept in item.json) the copy stays in the model's units (about one unit long).
--if-new: do nothing when item.json already names a model (the studio runs this after every TRELLIS run of an item, so
only the first one is chosen by itself; "Use for the item" chooses another).

Writes item.json: model, grip, length_units, length_m, sockets (default ["hand"]), game {model, faces, texture}.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

from reference.reproject import is_reprojected, paint_trimesh
from rigging.item_grip import estimate, vertices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("item")
    ap.add_argument("model")
    ap.add_argument("--faces", type=int, default=5000)
    ap.add_argument("--texture", type=int, default=1024)
    ap.add_argument("--length-m", type=float)
    ap.add_argument("--length-from")
    ap.add_argument("--if-new", action="store_true")
    a = ap.parse_args()
    d, model = Path(a.item).resolve(), Path(a.model).resolve()
    meta = json.loads((d / "item.json").read_text()) if (d / "item.json").exists() else {}
    if a.if_new and meta.get("model"):
        print(f"{d.name}: already uses {meta['model']}: kept (choose another with \"Use for the item\")")
        return
    sample = model.parent / "sample_xpu.pt"
    if not sample.exists():
        raise SystemExit(f"no {sample}: the run's voxels are needed to bake the light copy")
    r = lambda x: [round(float(t), 5) for t in x]
    name = str(model.relative_to(d))
    g = meta.get("grip")
    if not (g and g.get("estimated") == "by hand" and meta.get("model") == name):
        e = estimate(vertices(model))
        g = {"origin": r(e["origin"]), "x": r(e["x"]), "y": r(e["y"]), "z": r(e["z"]), "estimated": e["how"]}
        meta["length_units"] = round(e["length"], 4)
        print(f"{d.name}: grip by its {e['how']}")
    if a.length_m:
        meta.update(length_m=a.length_m, length_from=a.length_from or "given")
    meta.update(model=name, grip=g)
    meta.setdefault("sockets", ["hand"])

    from o_voxel import postprocess_torch
    v = torch.load(sample, weights_only=True)
    mesh = postprocess_torch.to_glb(vertices=v["vertices"], faces=v["faces"], attr_volume=v["attrs"],
                                    coords=v["coords"], attr_layout={k: slice(*s) for k, s in v["layout"].items()},
                                    voxel_size=v["voxel_size"], aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                    decimation_target=a.faces, texture_size=a.texture, verbose=True)
    painted = None
    if is_reprojected(model):  # the voxels have TRELLIS's colours, not the views': paint the copy as the model was
        painted = paint_trimesh(mesh, model.parent)
        print(f"{d.name}: light copy painted from the views: " + (f"{painted:.0%} of its texture" if painted is not None
              else "the run has none to paint from"))
    F = np.eye(4)
    F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = g["x"], g["y"], g["z"], g["origin"]
    k = meta["length_m"] / meta["length_units"] if meta.get("length_m") else 1.0
    mesh.apply_transform(np.diag([k, k, k, 1]) @ np.linalg.inv(F))
    out = d / "game" / f"{d.name}.glb"
    out.parent.mkdir(exist_ok=True)
    mesh.export(out, extension_webp=True)
    meta["game"] = {"model": str(out.relative_to(d)), "faces": int(len(mesh.faces)), "texture": a.texture,
                    "metres": bool(meta.get("length_m")), "painted_from_views": painted is not None}
    (d / "item.json").write_text(json.dumps(meta, indent=1))
    print(f"{d.name}: {out} ({len(mesh.faces)} faces, {a.texture} texture"
          + (f", {meta['length_m']} m long)" if meta.get("length_m") else ", in the model's units: no real length given)"))


if __name__ == "__main__":
    main()
