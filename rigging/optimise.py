"""An optimised copy of a TRELLIS model, as a model of its own: fewer faces, triangles or quads, its texture baked again.

    PYTHONPATH=external/trellis2/o-voxel python -m rigging.optimise MODEL.glb [--faces 30000] [--mode tris|quads]
        [--texture 2048] [--out DIR]

MODEL.glb is a TRELLIS run's model; the run's sample_xpu.pt (its mesh and the voxels with their colours) is what is
optimised, so nothing is lost to the model's own simplification. As postprocess_xpu.py, the surface is rebuilt from
the voxels and snapped onto TRELLIS's mesh (o_voxel's remesh_snap); then rigging/blender/optimise_mesh.py reduces it:
- tris: decimated to --faces triangles;
- quads: a quad layout (QuadriFlow) of about --faces / 2 quads. Even quads that follow the surface, but one density
  everywhere: where layers lie closer together than a quad (hair over a cheek) they fold.
Then a UV unwrap and the texture baked again from the voxels (o_voxel's bake_textured), --texture square. When MODEL.glb
is a reprojected one (reference/reproject.py: <name>_reprojected.glb), the baked texture is painted from the run's views
again, as the voxels hold only TRELLIS's own colours.

Writes DIR (default: beside the run, <run>_opt<faces>k[_quads]) with <name>_textured.glb (the studio's model, every face
a triangle, as glTF has no quads), layout.png (rigging/blender/layout_sheet.py: the faces drawn over the body and the
head) and optimise.json; for quads also quads.obj, .mtl and .png (the quads as quads, textured, for Blender).
The copy is a model like any other: compare it with the original, and use it for the rig.
"""
import argparse, json, os, subprocess, tempfile, time
from pathlib import Path

import numpy as np
import torch
from o_voxel.postprocess_torch import bake_textured, remesh_snap

from reference.reproject import is_reprojected, paint_trimesh

ROOT = Path(__file__).resolve().parent.parent
BLENDER = os.environ.get("BLENDER", "blender")


def triangles(v, loop_start, loop_total, loop_vert, loop_uv):
    """Each face split into triangles (a quad along its shorter diagonal, anything else as a fan): (T, 3) vertex
    indices and (T, 3, 2) corner UVs."""
    tri_loops = []
    q = loop_total == 4
    if q.any():
        s = loop_start[q][:, None] + np.arange(4)
        a, b, c, d = (v[loop_vert[s[:, i]]] for i in range(4))
        short = np.linalg.norm(a - c, axis=1) <= np.linalg.norm(b - d, axis=1)
        t1 = np.where(short[:, None], s[:, [0, 1, 2]], s[:, [1, 2, 3]])
        t2 = np.where(short[:, None], s[:, [0, 2, 3]], s[:, [1, 3, 0]])
        tri_loops += [t1, t2]
    for st, n in zip(loop_start[~q], loop_total[~q]):
        tri_loops.append(np.array([[st, st + i, st + i + 1] for i in range(1, n - 1)]))
    L = np.concatenate(tri_loops)
    return loop_vert[L], loop_uv[L]


def write_obj(out, v, loop_start, loop_total, loop_vert, loop_uv, texture):
    """The faces as made (quads), y up like the GLB, textured with the base colour."""
    texture.convert("RGB").save(out / "quads.png")
    (out / "quads.mtl").write_text("newmtl body\nKd 1 1 1\nmap_Kd quads.png\n")
    y_up = np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)
    with open(out / "quads.obj", "w") as f:
        f.write("mtllib quads.mtl\no body\n")
        f.writelines(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in y_up)
        f.writelines(f"vt {u:.6f} {w:.6f}\n" for u, w in loop_uv)
        f.write("usemtl body\n")
        for st, n in zip(loop_start, loop_total):
            f.write("f " + " ".join(f"{loop_vert[i] + 1}/{i + 1}" for i in range(st, st + n)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--faces", type=int, default=30000)
    ap.add_argument("--mode", choices=["tris", "quads"], default="tris")
    ap.add_argument("--texture", type=int, default=2048)
    ap.add_argument("--out")
    a = ap.parse_args()
    t0 = time.time()
    model = Path(a.model).resolve()
    run = model.parent
    sample = run / "sample_xpu.pt"
    if not sample.exists():
        raise SystemExit(f"no {sample}: only a TRELLIS run's model can be optimised (its voxels are baked again)")
    if a.out:
        out = Path(a.out).resolve()
    else:
        base = run.parent / f"{run.name}_opt{round(a.faces / 1000)}k" + ("_quads" if a.mode == "quads" else "")
        out, n = base, 2
        while out.exists():
            out, n = base.with_name(f"{base.name}_{n}"), n + 1
    out.mkdir(parents=True, exist_ok=True)

    d = torch.load(sample, weights_only=True)
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32)
    voxel_size = torch.as_tensor(np.broadcast_to(np.asarray(d["voxel_size"], np.float32), (3,)).copy())
    grid_size = ((aabb[1] - aabb[0]) / voxel_size).round().int()
    layout = {k: slice(*s) for k, s in d["layout"].items()}
    v, f = remesh_snap(d["vertices"], d["faces"], d["coords"], aabb, voxel_size, grid_size, 1e-5, 1e-4, "cpu", True, t0)

    with tempfile.TemporaryDirectory() as tmp:
        np.savez(f"{tmp}/in.npz", vertices=v, faces=f)
        p = subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(ROOT / "rigging/blender/optimise_mesh.py"),
                            "--", f"{tmp}/in.npz", f"{tmp}/out.npz", a.mode, str(a.faces), str(2 / a.texture)],
                           capture_output=True, text=True)
        if p.returncode or not os.path.exists(f"{tmp}/out.npz"):
            raise SystemExit(f"Blender failed:\n{p.stdout[-3000:]}\n{p.stderr[-2000:]}")
        print("".join(f"[{time.time() - t0:7.1f}s] {l}\n" for l in p.stdout.splitlines()
                      if l.startswith(("quads:", "tris:"))), end="", flush=True)
        r = dict(np.load(f"{tmp}/out.npz"))
    tv, tf_uv = triangles(r["vertices"], r["loop_start"], r["loop_total"], r["loop_vert"], r["loop_uv"])
    print(f"[{time.time() - t0:7.1f}s] {len(r['loop_start'])} faces as made, {len(tv)} triangles", flush=True)
    mesh = bake_textured(r["vertices"].astype(np.float64), tv, tf_uv, d["attrs"], d["coords"], layout, aabb, voxel_size,
                         grid_size, a.texture, True, t0)
    painted = None
    if is_reprojected(model):  # the voxels have TRELLIS's colours, not the views': paint the new texture as the model was
        painted = paint_trimesh(mesh, run)
        print(f"[{time.time() - t0:7.1f}s] painted from the views: " + (f"{painted:.0%} of the texture" if painted is not None
              else "the run has none to paint from"), flush=True)
    glb = out / f"{run.parent.name}_textured.glb"
    mesh.export(glb, extension_webp=True)
    if a.mode == "quads":
        write_obj(out, r["vertices"], r["loop_start"], r["loop_total"], r["loop_vert"], r["loop_uv"],
                  mesh.visual.material.baseColorTexture)
    sheet = out / "layout.png"
    subprocess.run([BLENDER, "-b", "--factory-startup", "-P", str(ROOT / "rigging/blender/layout_sheet.py"), "--",
                    str(out / "quads.obj" if a.mode == "quads" else glb), str(sheet)], check=True,
                   stdout=subprocess.DEVNULL)
    sizes = r["loop_total"]
    info = {"from": str(model.relative_to(run.parent.parent)) if run.parent.parent in model.parents else str(model),
            "mode": a.mode, "target_faces": a.faces, "triangles": int(len(tv)), "faces_as_made": int(len(sizes)),
            "quads": int((sizes == 4).sum()), "texture": a.texture, "painted_from_views": painted is not None, "seconds": round(time.time() - t0, 1)}
    (out / "optimise.json").write_text(json.dumps(info, indent=1))
    print(f"optimised: {info['triangles']} triangles" + (f" ({info['quads']} quads)" if a.mode == "quads" else "")
          + f", {a.texture} texture, {glb.stat().st_size / 1e6:.1f} MB -> {glb} (layout: {sheet})", flush=True)


if __name__ == "__main__":
    main()
