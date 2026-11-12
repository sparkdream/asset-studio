"""Put another model's texture on a character's rig, without rigging again: for a painted copy of the rig's own model
(reference/reproject.py: <name>_reprojected.glb), which has the same mesh and UVs and only a new base colour.

    python -m rigging.retexture PROJECT_DIR MODEL.glb [--dry-run]

MODEL.glb must have the rig's mesh: the same triangles and UVs as anymove/model.glb (the rig's copy of the model it was
made from). Its base colour image then replaces the current one wherever the rig holds that image: anymove/model.glb,
the blends under anymove/ and rigging/ (prep's static.blend, rigged.blend: rigging/blender/swap_texture.py), each
capture's GLB, and the game export's copy of the character (export/characters/<name>/). Found by the image's own bytes,
so nothing else changes and nothing that holds another texture is touched; the archive/ is left as it was.

A texture is no new rig or capture: every file keeps its modified time, so the studio doesn't take the rig as rebuilt,
the captures as stale or their checks as out of date. Videos already rendered (showcase, overlay) keep the old paint.
Kept as WebP in the GLBs (the paint is encoded at quality 90 if MODEL holds a PNG). rigging/promote.json records the
texture's model. Running it again with the plain model puts its texture back.
"""
import argparse, io, json, os, subprocess, tempfile, time
from pathlib import Path

import numpy as np
from PIL import Image

from reference.reproject import base_colour_image, join_glb, read_glb, split_glb

ROOT = Path(__file__).resolve().parent.parent
BLENDER = os.environ.get("BLENDER", "blender")


def image_bytes(path):
    """(bytes, MIME type) of a GLB's base colour image."""
    doc, binary = split_glb(Path(path).read_bytes())
    im = doc["images"][base_colour_image(doc)]
    v = doc["bufferViews"][im["bufferView"]]
    s = v.get("byteOffset", 0)
    return binary[s:s + v["byteLength"]], im.get("mimeType")


def swap_image(data, old, new):
    """The GLB with every image whose bytes are `old` given `new` (same format), buffer views repacked; None if it holds
    no such image."""
    doc, binary = split_glb(data)
    views = {im["bufferView"] for im in doc.get("images", []) if "bufferView" in im}
    chunk = lambda v: binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
    hits = {i for i in views if chunk(doc["bufferViews"][i]) == old}
    if not hits:
        return None
    packed = b""
    for i, v in enumerate(doc["bufferViews"]):
        c = new if i in hits else chunk(v)
        packed += b"\0" * (-len(packed) % 4)
        v["byteOffset"], v["byteLength"] = len(packed), len(c)
        packed += c
    doc["buffers"][0]["byteLength"] = len(packed)
    return join_glb(doc, packed)


def keep_time(path, write):
    """Run write() on path, then give it back its modified time."""
    st = path.stat()
    write()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("model")
    ap.add_argument("--dry-run", action="store_true", help="list what would change")
    a = ap.parse_args()
    t0 = time.time()
    p, model = Path(a.project).resolve(), Path(a.model).resolve()
    rig_copy = p / "anymove/model.glb"
    if not rig_copy.exists():
        raise SystemExit(f"{p.name} has no rig (no {rig_copy})")
    _, uv_a, f_a, _ = read_glb(rig_copy)
    _, uv_b, f_b, _ = read_glb(model)
    if f_a.shape != f_b.shape or not (f_a == f_b).all() or uv_a.shape != uv_b.shape or np.abs(uv_a - uv_b).max() > 1e-6:
        raise SystemExit(f"{model.name} is not the rig's mesh (other triangles or UVs): only a copy of the rig's own model "
                         "with another paint can go on it; use it for the rig instead (rigging/promote.py)")
    old, old_mime = image_bytes(rig_copy)
    new, mime = image_bytes(model)
    if old_mime != "image/webp":
        raise SystemExit(f"the rig's texture is {old_mime}, not WebP: not a rig this was made for")
    if mime != "image/webp":
        buf = io.BytesIO()
        Image.open(io.BytesIO(new)).save(buf, format="WEBP", quality=90, method=6)
        new = buf.getvalue()
    if new == old:
        print(f"{p.name}: the rig already wears {model.name}'s texture")
        return

    glbs = [g for d in (p / "anymove", p / "rigging", p.parent / "export/characters" / p.name) if d.is_dir()
            for g in sorted(d.rglob("*.glb"))]
    blends = [b for d in (p / "anymove", p / "rigging") if d.is_dir() for b in sorted(d.rglob("*.blend"))]
    changed = []
    for g in glbs:
        out = swap_image(g.read_bytes(), old, new)
        if out is not None:
            changed.append(g)
            if not a.dry_run:
                keep_time(g, lambda: g.write_bytes(out))
    print(f"GLBs: {len(changed)} of {len(glbs)} wore the old texture" + (" (dry run)" if a.dry_run else ": swapped"))
    for g in changed:
        print("  " + str(g.relative_to(p.parent)))
    if not a.dry_run and blends:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "old.webp").write_bytes(old)
            Path(tmp, "new.webp").write_bytes(new)
            for b in blends:
                def run():
                    r = subprocess.run([BLENDER, "-b", "--factory-startup", str(b), "-P",
                                        str(ROOT / "rigging/blender/swap_texture.py"), "--", f"{tmp}/old.webp",
                                        f"{tmp}/new.webp"], capture_output=True, text=True)
                    lines = [l for l in r.stdout.splitlines() if l.startswith("swap_texture:")]
                    if r.returncode or not lines:
                        raise SystemExit(f"Blender failed on {b}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                    print(f"  {b.relative_to(p.parent)}: {lines[-1].split(':', 1)[1].strip()}")
                keep_time(b, run)
    if not a.dry_run:
        rec_path = p / "rigging/promote.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
        name = model.relative_to(p).as_posix() if p in model.parents else str(model)
        if name == rec.get("model"):
            rec.pop("texture", None)
            rec.pop("texture_when", None)
        else:
            rec.update(texture=name, texture_when=time.strftime("%Y-%m-%d %H:%M"))
        if rec_path.parent.is_dir():
            rec_path.write_text(json.dumps(rec, indent=1))
    print(f"{p.name}: the rig wears {model.name}'s texture ({time.time() - t0:.0f}s)" if not a.dry_run else "")


if __name__ == "__main__":
    main()
