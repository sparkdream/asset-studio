"""Paint a TRELLIS.2 model's base colour with the real pixels of the reference views it was made from.

    python -m reference.reproject RUN/MODEL.glb [--views front,back] [--out OUT.glb] [--weights]

TRELLIS redraws the picture: a face, a trim pattern, a buckle come back as blurry lookalikes, because its texture lives on
a coarse voxel grid (32^3 at 512). The views already hold the right answer for every surface they can see, so each such
surface takes its view's pixel, and the model keeps its own paint only where no view looks. The paint the views don't
reach is shifted towards theirs, spot by spot, so the two meet without a seam.

Views: by default the front and back, of those the run was made from (RUN/views.json: the set and each view's version,
from the project's views/ or views/history/); --views names others. A run made from one image paints from that image
as its front view (the outline fit gives it no trust if it is not one). A run with no view that fits writes nothing.
A view is placed by its name (reference/views.py: front, front_left, left ... at 45 degree steps), looking
orthographically at the model in TRELLIS's frame (z up, the subject facing -y; see
external/trellis2/trellis2/pipelines/view_weights.py). Generated views carry no camera, so each one's cut-out (BiRefNet,
as TRELLIS cuts its inputs; cached in RUN/reproject/) is scaled and shifted onto the model's outline from that angle,
then its features (eyes, mouth, trim) are moved a few pixels onto where the model built them (checked optical flow).

A texel takes a view's pixel only when the view sees it (nothing in front), it faces the view (grazing angles fade out),
it is away from the model's outline, the cut-out's edge and inner edges (one part in front of another), and it is inside
the cut-out. Only the base colour image changes; mesh, UVs and every other byte of the GLB stay.

The studio runs this after every TRELLIS run ("Paint from views", on by default). Copies made again from a run's voxels
(rigging/optimise.py, rigging/item_game.py) are painted too when made of a _reprojected model (paint_trimesh).

An optimised copy (rigging/optimise.py) is painted from the views of the run it was made from (its optimise.json).

Writes OUT (default <model>_reprojected.glb beside the model) and, with --weights, reproject/<model>_trust.png beside it: white where a
view's pixel was used, black where the model kept its own paint (the first thing to look at when a result looks wrong).

Only as good as the views' agreement with the model. On the rogue (rogue_apose, trellis_multi) the front and back
outlines fit at IoU 0.90 (median gap 1.4 px at 768); the generated side and three-quarter views at 0.70-0.83 (5-7 px),
and a three-quarter view painted the face twice. So those are opt-in, a view's trust falls with its outline fit (printed
per view), and where views overlap the squarest one wins rather than an average. A generated view also carries its own
colour drift (the rogue's back view has lighter hair than her front) and its lighting into the base colour.

Adapted from image-to-3dlab (https://github.com/Bingeljell/image-to-3dlab, Apache-2.0, Copyright 2026 Bingeljell and
image-to-3dlab contributors): image_to_3dlab/photo_paint.py (rasterizer, trust weights, local colour match, GLB texture
swap) and image_to_3dlab/view_align.py (feature nudge, now optical flow). Changed here: orthographic cameras placed by view name instead
of saved perspective cameras, a wider outline fit (generated views have no camera), BiRefNet cut-outs, view versions.
"""
import argparse, io, json, struct, sys, time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ANGLES = {"front": 0, "front_left": 45, "left": 90, "back_left": 135, "back": 180, "back_right": 225, "right": 270,
          "front_right": 315}


@dataclass(frozen=True)
class Settings:
    facing_low: float = 0.25    # dot(normal, towards the camera): no trust below, full trust from facing_high
    facing_high: float = 0.6
    normal_rounds: int = 8      # neighbour averaging of the normals the facing test uses (see smooth_normals)
    edge_px: int = 6            # fade in from the model's outline over this many pixels (pixel sizes are for a
    cutout_edge: int = 8        # 768 px tall grid and scale with it); from the cut-out's own edge (half backdrop);
    inner_edge: int = 8         # beside inner edges (an ear in front of a head; 0 turns it off)
    inner_jump: float = 0.02    # a depth step bigger than this share of the model's size is an inner edge
    depth_tolerance: float = 0.004  # seen: within this share of the model's size of the nearest surface
    gutter_px: int = 4          # grow painted texels into the UV gutter so filtering doesn't pull old paint back
    local_colour: bool = True   # shift the unseen paint towards the views spot by spot (else one shift overall)
    nudge: bool = True          # move each view's features onto where the model built them
    fit_floor: float = 0.88     # outline IoU below which a view's trust is scaled down (to 0 at fit_zero: the outlines
    fit_zero: float = 0.8       # don't really agree, and painting from it would put features in the wrong place)
    overlap_power: float = 8.0  # where views overlap, each counts as its trust to this power: the squarest view wins
                                # instead of an average, since two pictures a few pixels apart average into ghosts


# --- GLB in and out (from photo_paint.py) ------------------------------------------------------------------------------

def split_glb(data):
    if data[:4] != b"glTF":
        raise ValueError("not a GLB file")
    n = struct.unpack_from("<I", data, 12)[0]
    doc, off = json.loads(data[20:20 + n]), 20 + n
    binary = data[off + 8:off + 8 + struct.unpack_from("<I", data, off)[0]] if off < len(data) else b""
    return doc, binary


def join_glb(doc, binary):
    text = json.dumps(doc, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    binary += b"\0" * (-len(binary) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(text) + (8 + len(binary) if binary else 0))
    out += struct.pack("<II", len(text), 0x4E4F534A) + text
    return out + (struct.pack("<II", len(binary), 0x004E4942) + binary if binary else b"")


def accessor(doc, binary, i):
    a = doc["accessors"][i]
    v = doc["bufferViews"][a["bufferView"]]
    dtype = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}[a["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]
    if v.get("byteStride") and v["byteStride"] != np.dtype(dtype).itemsize * width:
        raise ValueError("interleaved buffers are not supported")
    return np.frombuffer(binary, dtype=dtype, count=a["count"] * width,
                         offset=v.get("byteOffset", 0) + a.get("byteOffset", 0)).reshape(a["count"], width)


def base_colour_image(doc):
    mats = doc.get("materials") or []
    if len(mats) != 1:
        raise ValueError(f"expected one material, found {len(mats)}")
    tex = mats[0].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if tex is None:
        raise ValueError("the material has no base colour texture")
    entry = doc["textures"][tex["index"]]
    return entry["source"] if "source" in entry else entry["extensions"]["EXT_texture_webp"]["source"]


def read_glb(path):
    """Positions, glTF UVs (0,0 top left), triangles and the base colour image of a one-mesh GLB with no transforms."""
    doc, binary = split_glb(Path(path).read_bytes())
    if any(k in n for n in doc.get("nodes", []) for k in ("matrix", "translation", "rotation", "scale")):
        raise ValueError("node transforms are not supported (a TRELLIS run's GLB has none)")
    prims = [p for m in doc["meshes"] for p in m["primitives"]]
    if len(prims) != 1:
        raise ValueError(f"expected one primitive, found {len(prims)}")
    at = prims[0]["attributes"]
    pos = accessor(doc, binary, at["POSITION"]).astype(np.float64)
    uv = accessor(doc, binary, at["TEXCOORD_0"]).astype(np.float64)
    faces = accessor(doc, binary, prims[0]["indices"]).reshape(-1, 3).astype(np.int64)
    v = doc["bufferViews"][doc["images"][base_colour_image(doc)]["bufferView"]]
    s = v.get("byteOffset", 0)
    pic = Image.open(io.BytesIO(binary[s:s + v["byteLength"]]))
    return pos, uv, faces, np.asarray(pic.convert("RGBA" if "A" in pic.getbands() else "RGB"))


def replace_base_colour(data, png):
    """The GLB with its base colour image swapped for `png`; buffer views repacked, everything else unchanged."""
    doc, binary = split_glb(data)
    img = base_colour_image(doc)
    target = doc["images"][img]["bufferView"]
    packed = b""
    for i, v in enumerate(doc["bufferViews"]):
        chunk = png if i == target else binary[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]
        packed += b"\0" * (-len(packed) % 4)
        v["byteOffset"], v["byteLength"] = len(packed), len(chunk)
        packed += chunk
    doc["buffers"][0]["byteLength"] = len(packed)
    doc["images"][img]["mimeType"] = "image/png"
    for t in doc.get("textures", []):  # the image is a PNG now: point straight at it rather than through the WebP extension
        ext = t.get("extensions", {})
        if ext.get("EXT_texture_webp", {}).get("source") == img:
            t["source"] = img
            del ext["EXT_texture_webp"]
            if not ext:
                t.pop("extensions")
    if not any("EXT_texture_webp" in t.get("extensions", {}) for t in doc.get("textures", [])):
        for k in ("extensionsUsed", "extensionsRequired"):
            if k in doc:
                doc[k] = [n for n in doc[k] if n != "EXT_texture_webp"]
                if not doc[k]:
                    del doc[k]
    return join_glb(doc, packed)


# --- geometry (from photo_paint.py) ------------------------------------------------------------------------------------

def rasterize(tri_xy, tri_depth, size, chunk=4096):
    """Triangles (F,3,2 pixel coords; F,3 depth) into a (h, w) grid, nearest first: triangle index per pixel (-1 none)
    and the barycentric weights at each pixel centre (pixel centres at +0.5)."""
    h, w = size
    best = np.full(h * w, np.inf)
    face_of = np.full(h * w, -1, dtype=np.int64)
    bary = np.zeros((h * w, 3))
    for start in range(0, len(tri_xy), chunk):
        xy, dz = tri_xy[start:start + chunk], tri_depth[start:start + chunk]
        lo = np.clip(np.floor(xy.min(1) - 0.5), 0, [w - 1, h - 1]).astype(np.int64)
        hi = np.clip(np.ceil(xy.max(1) - 0.5), 0, [w - 1, h - 1]).astype(np.int64)
        ext = hi - lo + 1
        counts = ext[:, 0] * ext[:, 1]
        counts[(xy.max(1) < 0).any(1) | (xy.min(1)[:, 0] > w) | (xy.min(1)[:, 1] > h)] = 0
        if counts.sum() == 0:
            continue
        tri = np.repeat(np.arange(len(xy)), counts)
        off = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        px, py = lo[tri, 0] + off % ext[tri, 0], lo[tri, 1] + off // ext[tri, 0]
        cx, cy = px + 0.5, py + 0.5
        a, b, c = xy[tri, 0], xy[tri, 1], xy[tri, 2]
        den = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1.0)
        w0 = ((b[:, 1] - c[:, 1]) * (cx - c[:, 0]) + (c[:, 0] - b[:, 0]) * (cy - c[:, 1])) / den
        w1 = ((c[:, 1] - a[:, 1]) * (cx - c[:, 0]) + (a[:, 0] - c[:, 0]) * (cy - c[:, 1])) / den
        w2 = 1.0 - w0 - w1
        inside = ok & (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        tri, w0, w1, w2 = tri[inside], w0[inside], w1[inside], w2[inside]
        pixel = py[inside] * w + px[inside]
        z = w0 * dz[tri, 0] + w1 * dz[tri, 1] + w2 * dz[tri, 2]
        order = np.lexsort((z, pixel))
        pixel, z, tri, w0, w1, w2 = (v[order] for v in (pixel, z, tri, w0, w1, w2))
        first = np.ones(len(pixel), bool)
        first[1:] = pixel[1:] != pixel[:-1]
        pixel, z, tri, w0, w1, w2 = (v[first] for v in (pixel, z, tri, w0, w1, w2))
        nearer = z < best[pixel]
        pixel = pixel[nearer]
        best[pixel] = z[nearer]
        face_of[pixel] = tri[nearer] + start
        bary[pixel] = np.stack([w0[nearer], w1[nearer], w2[nearer]], 1)
    return face_of.reshape(h, w), bary.reshape(h, w, 3)


def vertex_normals(pos, faces):
    """Area-weighted vertex normals (up to sign: generated meshes may be wound either way; see outward_sign)."""
    tri = pos[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n = np.zeros_like(pos)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def weld(pos):
    """Vertices split by UV seams merged back: an index per vertex."""
    return np.unique(np.round(pos, 6), axis=0, return_inverse=True)[1].reshape(-1)


def smooth_normals(pos, faces, normals, rounds):
    """Vertex normals averaged with their neighbours' `rounds` times, across UV seams. TRELLIS's surface is bumpy at
    the voxel scale, and the facing test on raw normals scatters dots of low trust over surfaces that face the view
    squarely; whether a surface grazes the view is a question at the scale of the view's pixels, not of the bumps."""
    wv = weld(pos)
    count, corners = wv.max() + 1, wv[faces]
    e = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
    adj = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(count, count))
    adj = (((adj + adj.T) > 0).astype(np.float64) + diags(np.ones(count))).tocsr()
    adj = diags(1.0 / np.asarray(adj.sum(1)).ravel()) @ adj
    n = np.zeros((count, 3))
    np.add.at(n, wv, normals)
    for _ in range(rounds):
        n = adj @ n
    n = n[wv]
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def outward_sign(pos, faces, open_share=0.02):
    """+1 if vertex_normals point out of the mesh, -1 if it is inside out, 0 if it is open (no inside to tell)."""
    wf = weld(pos)[faces]
    edges = np.sort(np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]]), axis=1)
    if (np.unique(edges, axis=0, return_counts=True)[1] == 1).mean() > open_share:
        return 0.0
    c = pos - pos.mean(0)
    vol = np.einsum("ij,ij->i", c[faces[:, 0]], np.cross(c[faces[:, 1]], c[faces[:, 2]])).sum()
    return float(np.sign(vol))


def erode(mask, pixels):
    """Whole-pixel distance from each True pixel to the nearest False, capped at `pixels` (0 outside)."""
    if pixels <= 0:
        return mask.astype(np.int32)
    return np.minimum(ndi.distance_transform_cdt(np.pad(mask, 1), metric="taxicab")[1:-1, 1:-1], pixels)


def inner_edges(zbuf, jump):
    """Pixels where the surface steps in depth by more than `jump` to a neighbour that is also surface."""
    seen = np.isfinite(zbuf)
    d = np.where(seen, zbuf, 0.0)
    cut = np.zeros(zbuf.shape, bool)
    for here, there in (((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                        ((slice(1, None), slice(None)), (slice(None, -1), slice(None)))):
        step = seen[here] & seen[there] & (np.abs(d[here] - d[there]) > jump)
        cut[here] |= step
        cut[there] |= step
    return cut


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def sample_rgb(image, x, y):
    """Bilinear RGB at pixel coordinates (centres at +0.5), clamped."""
    h, w = image.shape[:2]
    fx, fy = np.clip(x - 0.5, 0, w - 1), np.clip(y - 0.5, 0, h - 1)
    x0, y0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    tx, ty = (fx - x0)[:, None], (fy - y0)[:, None]
    rgb = image[..., :3].astype(np.float64)
    return (rgb[y0, x0] * (1 - tx) + rgb[y0, x1] * tx) * (1 - ty) + (rgb[y1, x0] * (1 - tx) + rgb[y1, x1] * tx) * ty


# --- our cameras: orthographic, placed by the view's name ---------------------------------------------------------------

def trellis_frame(pos):
    """glTF (y up) -> TRELLIS's frame (z up, the subject facing -y), the frame view angles are measured in."""
    return np.stack([pos[:, 0], -pos[:, 2], pos[:, 1]], 1)


class Camera:
    """Orthographic, from angle a (degrees round the subject towards its own left): the camera stands along
    (sin a, -cos a, 0), image right is (cos a, sin a, 0), image up is z. `px` metres per pixel; the grid frames `pos`
    with a margin."""

    def __init__(self, angle, pos, px, margin=16):
        a = np.radians(angle)
        self.towards = np.array([np.sin(a), -np.cos(a), 0.0])  # from the subject to the camera
        self.right = np.array([np.cos(a), np.sin(a), 0.0])
        u, v = pos @ self.right, pos[:, 2]
        self.px, self.u0, self.v0 = px, u.min() - margin * px, v.max() + margin * px
        self.size = (int(np.ceil((v.max() - v.min()) / px)) + 2 * margin, int(np.ceil((u.max() - u.min()) / px)) + 2 * margin)

    def project(self, p):
        """Points -> pixel x, y and depth (smaller is nearer the camera)."""
        return (p @ self.right - self.u0) / self.px, (self.v0 - p[:, 2]) / self.px, -(p @ self.towards)

    def render(self, pos, faces):
        """Which triangle each pixel sees, with barycentrics, and the depth buffer (inf where nothing)."""
        x, y, z = self.project(pos)
        face_of, bary = rasterize(np.stack([x, y], 1)[faces], z[faces], self.size)
        zbuf = np.full(self.size, np.inf)
        seen = face_of >= 0
        zbuf[seen] = (bary[seen] * z[faces][face_of[seen]]).sum(1)
        return face_of, bary, zbuf


# --- views: cut out, fitted onto the model ------------------------------------------------------------------------------

_birefnet = None


def cutout(path, cache):
    """The view as RGBA, its alpha from BiRefNet (the model TRELLIS cuts its inputs with); cached as a PNG."""
    global _birefnet
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.asarray(Image.open(cache).convert("RGBA"))
    im = Image.open(path)
    if im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 250:  # already cut out
        rgba = np.asarray(im)
    else:
        if _birefnet is None:
            import torch
            sys.path.insert(0, str(ROOT / "external/trellis2"))
            from trellis2.pipelines.rembg.BiRefNet import BiRefNet
            _birefnet = BiRefNet()
            dev = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
            _birefnet.to(dev)
            print(f"cut-outs: BiRefNet on {dev}", flush=True)
        rgba = np.asarray(_birefnet(im.convert("RGB")))
    cache.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(cache)
    return rgba


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def fit_outline(model, picture):
    """Scale and shift taking the picture's cut-out (any size) onto the model's outline (the camera grid): the box fit
    first (height and centre), then a local search for the best overlap, coarse to fine. Returns the 2x3 matrix taking
    grid pixels to picture pixels, and the overlap before (box fit) and after."""
    ys, xs = np.nonzero(picture)
    my, mx = np.nonzero(model)
    s = (ys.max() - ys.min()) / max(my.max() - my.min(), 1)  # picture pixels per grid pixel
    tx, ty = xs.mean() - mx.mean() * s, ys.min() - my.min() * s

    def warp(s, tx, ty, shrink=1):
        m = np.array([[s, 0, tx], [0, s, ty]], np.float32)
        if shrink > 1:  # grid pixels shrink x coarser
            m[:, :2] *= shrink
        h, w = model.shape[0] // shrink, model.shape[1] // shrink
        return cv2.warpAffine(picture.astype(np.uint8), m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) > 0

    def score(p, shrink):
        return iou(warp(*p, shrink), model[::shrink, ::shrink][:model.shape[0] // shrink, :model.shape[1] // shrink])
    p = (s, tx, ty)
    before = score(p, 1)
    for shrink, step in ((4, 8.0), (2, 2.0), (1, 1.0), (1, 0.5)):
        best = score(p, shrink)
        moved = True
        while moved:
            moved = False
            for d in ((step * s / 400, 0, 0), (-step * s / 400, 0, 0), (0, step * s, 0), (0, -step * s, 0),
                      (0, 0, step * s), (0, 0, -step * s)):
                q = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if (sc := score(q, shrink)) > best + 1e-6:
                    p, best, moved = q, sc, True
    return np.array([[p[0], 0, p[1]], [0, p[0], p[2]]]), before, score(p, 1)


def grey(rgb, mask):
    g = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
    if not mask.any():
        return np.full(g.shape, 128, np.uint8)
    g = np.where(mask, (g - g[mask].mean()) / (g[mask].std() + 1e-6) * 40 + 128, 128)
    return np.clip(g, 0, 255).astype(np.uint8)


def local_contrast(g, scale):
    """Detail only: each pixel's difference from its surroundings over their spread, so two pictures of one thing in
    other colours, lighting or blur (a view and TRELLIS's paint of it) look alike."""
    g = g.astype(np.float32)
    d = g - cv2.GaussianBlur(g, (0, 0), scale)
    return np.clip(d / (np.sqrt(cv2.GaussianBlur(d * d, (0, 0), scale * 2)) + 4) * 40 + 128, 0, 255).astype(np.uint8)


def match_features(own, picture, mask, max_px, detail_share=0.4, sigma_px=12, consistent_px=1.0, full=0.15):
    """Offsets (H,W,2: x, y) with own(p) ~ picture(p + offset): where the view drew a feature against where the model
    has it (TRELLIS's paint sits on its own geometry). Dense optical flow (DIS) between the two pictures' local
    contrast, kept only where it is sure: on detail (the most detailed `detail_share` of the figure), and where the
    flow back undoes the flow there (within consistent_px). The kept offsets are spread smoothly (sigma_px; fading
    to none away from kept ones, full where `full` of the neighbourhood was kept) and capped at max_px.
    Sizes are for a 768 px tall grid. Sparse confident patch matches (view_align.py's way) found a dozen spots on the
    rogue, all round the eyes, and their one smooth shift took her mouth up with her eyes, onto her upper lip; the
    flow moves each feature on its own (eyes 7 px, nose 3, mouth 2 at 1037 px, to within 1)."""
    h, w = mask.shape
    k = h / 768
    field = np.zeros((h, w, 2), np.float32)
    if mask.sum() < 100:
        return field, 0.0
    a, b = (local_contrast(grey(x, mask), max(2.0, 6 * k)) for x in (own, picture))
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fwd, bwd = dis.calc(a, b, None), dis.calc(b, a, None)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    err = np.linalg.norm(fwd + cv2.remap(bwd, gx + fwd[..., 0], gy + fwd[..., 1], cv2.INTER_LINEAR), axis=2)
    detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(grey(own, mask), (0, 0), 1.0), cv2.CV_32F)),
                              (0, 0), 6 * k)
    good = mask & (err < consistent_px * k) & (detail > np.percentile(detail[mask], 100 * (1 - detail_share)))
    wgt = good.astype(np.float32)
    total = cv2.GaussianBlur(fwd * wgt[..., None], (0, 0), sigma_px * k)
    hits = cv2.GaussianBlur(wgt, (0, 0), sigma_px * k)
    field = total / np.maximum(hits, 1e-6)[..., None] * np.clip(hits / full, 0, 1)[..., None]
    field *= np.minimum(1.0, max_px / np.maximum(np.linalg.norm(field, axis=2, keepdims=True), 1e-6))
    field[~mask] = 0
    return field.astype(np.float32), float(good[mask].mean())


@dataclass
class View:
    name: str
    cam: Camera
    image: np.ndarray  # RGBA on the camera's grid
    trust: float       # from how well its outline fits the model's


def place_view(name, rgba, pos, faces, uvs, texture, px, st):
    """A view's picture moved onto its camera's grid: the cut-out fitted to the model's outline, then detailed spots
    moved onto the model's own paint as the same camera sees it (match_features)."""
    cam = Camera(ANGLES[name], pos, px)
    face_of, bary, _ = cam.render(pos, faces)
    seen = face_of >= 0
    m, before, after = fit_outline(seen, rgba[..., 3] > 127)
    h, w = cam.size
    img = cv2.warpAffine(rgba, m.astype(np.float32), (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    n = 0.0
    if st.nudge:
        uv = (bary[seen][:, :, None] * uvs[faces[face_of[seen]]]).sum(1)
        th, tw = texture.shape[:2]
        own = np.zeros((h, w, 3), np.uint8)
        own[seen] = texture[np.clip((uv[:, 1] * th).astype(int), 0, th - 1), np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1), :3]
        field, n = match_features(own, img[..., :3], seen & (img[..., 3] > 127), 10 * h / 768)
        if n:
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            moved = cv2.remap(img, gx + field[..., 0], gy + field[..., 1], cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            moved[..., 3] = img[..., 3]  # colours move onto features; the cut-out stays fitted
            img = moved
    trust = float(smoothstep(after, st.fit_zero, st.fit_floor))
    print(f"{name}: outline fit {before:.3f} -> {after:.3f} (scale {1 / m[0, 0]:.3f} grid px per picture px), "
          f"features matched on {n:.0%} of it, trust {trust:.2f}", flush=True)
    return View(name, cam, img, trust)


# --- painting (from photo_paint.py) ------------------------------------------------------------------------------------

def view_weights(points, normals, view, pos, faces, outward, st):
    """How much to trust the view's pixel at each texel, and that pixel's colour."""
    cam, img = view.cam, view.image
    h, w = cam.size
    face_of, _, zbuf = cam.render(pos, faces)
    seen = face_of >= 0
    x, y, depth = cam.project(points)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy = np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)
    extent = np.ptp(pos, axis=0).max()
    visible = inside & (depth <= zbuf[iy, ix] + st.depth_tolerance * extent)
    facing = normals @ cam.towards
    # on a closed mesh only surfaces facing the camera take its paint (the far side of a thin coat flap or lock of hair
    # sits inside the depth tolerance); an open mesh has no outside to tell, so either side counts. But the frontmost
    # surface at its pixel is what any render from here shows, whichever way it faces: TRELLIS's mesh is a shell two
    # voxels thick (an outer surface facing out, an inner one facing in), and where the inner one pokes through, its
    # triangles are drawn; skipped, they kept their old paint and dotted the model (9% of the triangles the rogue's
    # front camera sees face away from it)
    frontmost = depth <= zbuf[iy, ix] + 0.5 * cam.px
    # and the shell's other layer just behind it, facing away: on a solid part that is its inner surface, which no
    # camera ever sees head on but which pokes through from other angles (left with its old paint, it speckled the
    # rogue's cheeks and hair from three-quarter views); on a thin part (a coat tail) it is the back face, which another
    # view sees. So it is a second choice: taken only where no view sees the texel facing it (see paint)
    behind = visible & ~frontmost & ((facing * outward < 0) if outward else np.zeros(len(facing), bool))
    second = np.abs(facing)
    facing = np.where(frontmost, np.abs(facing), np.maximum(facing * outward, 0.0) if outward else np.abs(facing))
    k = h / 768
    edge_px, inner_px, cut_px = (max(1, round(v * k)) for v in (st.edge_px, st.inner_edge, st.cutout_edge))
    edge = erode(seen, edge_px)[iy, ix] / edge_px
    if st.inner_edge > 0:
        edge = edge * smoothstep(erode(~inner_edges(zbuf, st.inner_jump * extent), inner_px)[iy, ix] / inner_px, 0, 1)
    matte = img[iy, ix, 3] / 255.0 * smoothstep(erode(img[..., 3] > 127, cut_px)[iy, ix] / cut_px, 0.0, 1.0)
    rest = smoothstep(edge, 0.0, 1.0) * matte * view.trust
    weight = visible * smoothstep(facing, st.facing_low, st.facing_high) * rest
    fallback = behind * smoothstep(second, st.facing_low, st.facing_high) * rest
    return weight, fallback, sample_rgb(img, x, y)


def local_colour_offset(pos, faces, texel_face, texel_bary, own, photo, weight, trusted=0.9, min_texels=500,
                        smooth=0.05, outlier=3.0, similar=20.0):
    """Per-texel colour offset taking the model's own paint towards the views, locally: the gap (view - own) measured
    where a view is fully trusted, put on the welded vertices and spread over the mesh by a smooth interpolation (a
    graph Laplacian holding the measured vertices loosely). A gap far from the typical one is a mismatched feature, not
    colour drift, and is left out so it can't smear. None when too little is trusted to measure.
    It spreads between vertices of like colour (the model's own paint, edges weighted exp(-(d / similar)^2) for RGB
    distance d): skin takes skin's offset, not an average with the hair beside it. Spread evenly, the rogue's jaw and
    the edge of a strand on her cheek (just off the views' trust, beside depth edges) kept a band of her old, warmer
    skin."""
    mask = weight >= trusted
    if mask.sum() < min_texels:
        return None
    gap = photo - own
    typical = np.median(gap[mask], axis=0)
    dev = np.linalg.norm(gap - typical, axis=1)
    mask &= dev <= max(np.median(dev[mask]) * 1.4826 * outlier, 1.0)
    wv = weld(pos)
    count, corners = wv.max() + 1, wv[faces]
    total, seen = np.zeros((count, 3)), np.zeros(count)
    for k in range(3):
        v, share = corners[texel_face[mask], k], texel_bary[mask, k]
        np.add.at(seen, v, share)
        np.add.at(total, v, share[:, None] * gap[mask])
    known = seen > 0.5
    target = np.zeros((count, 3))
    target[known] = total[known] / seen[known][:, None] - typical
    paint_sum, paint_n = np.zeros((count, 3)), np.zeros(count)  # each vertex's own colour, from all its texels
    for k in range(3):
        np.add.at(paint_n, corners[texel_face, k], texel_bary[:, k])
        np.add.at(paint_sum, corners[texel_face, k], texel_bary[:, k, None] * own)
    colour = paint_sum / np.maximum(paint_n, 1e-9)[:, None]
    e = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
    e = np.unique(np.sort(e[e[:, 0] != e[:, 1]], axis=1), axis=0)
    like = np.exp(-(np.linalg.norm(colour[e[:, 0]] - colour[e[:, 1]], axis=1) / similar) ** 2) + 1e-4
    adj = coo_matrix((like, (e[:, 0], e[:, 1])), shape=(count, count))
    adj = (adj + adj.T).tocsr()
    hold = known * smooth
    system = (diags(np.asarray(adj.sum(1)).ravel() + hold + 1e-6) - adj).tocsr()
    pre = diags(1.0 / system.diagonal())
    field = np.zeros((count, 3))
    for c in range(3):
        field[:, c], _ = cg(system, hold * target[:, c], M=pre, maxiter=3000, rtol=1e-6)
    return typical + (texel_bary[:, :, None] * field[corners[texel_face]]).sum(1)


def grow_into_gutter(image, weight, covered, steps):
    """Copy painted texels outward into the uncovered gutter, a ring at a time."""
    filled = covered & (weight > 0)
    for _ in range(steps):
        dist, (iy, ix) = ndi.distance_transform_edt(~filled, return_indices=True)
        ring = (dist > 0) & (dist <= 1.5) & ~covered
        image[ring, :3] = image[iy[ring], ix[ring], :3]
        weight[ring] = weight[iy[ring], ix[ring]]
        filled |= ring
        covered = covered | ring
    return image, weight


def paint(texture, gpos, uvs, faces, views, pos, st):
    """Blend the views' pixels into the base colour texture. Returns the new texture and the trust map."""
    h, w = texture.shape[:2]
    face_of, bary = rasterize(np.stack([uvs[:, 0] * w, uvs[:, 1] * h], 1)[faces], np.zeros((len(faces), 3)), (h, w))
    covered = face_of >= 0
    tf, tb = face_of[covered], bary[covered]
    normals = smooth_normals(pos, faces, vertex_normals(pos, faces), st.normal_rounds)
    points = (tb[:, :, None] * pos[faces[tf]]).sum(1)
    tn = (tb[:, :, None] * normals[faces[tf]]).sum(1)
    tn /= np.maximum(np.linalg.norm(tn, axis=1, keepdims=True), 1e-12)
    outward = outward_sign(pos, faces)
    total, colour, strongest = np.zeros(len(points)), np.zeros((len(points), 3)), np.zeros(len(points))
    per_view = [view_weights(points, tn, v, pos, faces, outward, st) for v in views]
    direct = np.max([w for w, _, _ in per_view], axis=0)
    for v, (wgt, fallback, rgb) in zip(views, per_view):
        wgt = np.maximum(wgt, fallback * (1 - direct))  # the layer behind: only what no view sees head on
        total += wgt ** st.overlap_power
        colour += (wgt ** st.overlap_power)[:, None] * rgb
        strongest = np.maximum(strongest, wgt)
        print(f"{v.name}: {(wgt > 0.5).mean():.0%} of the model's texels taken from it", flush=True)
    photo = colour / np.maximum(total, 1e-12)[:, None]
    out = texture.astype(np.float64).copy()
    flat = out.reshape(-1, out.shape[2])
    idx = np.flatnonzero(covered.ravel())
    local = local_colour_offset(gpos, faces, tf, tb, flat[idx, :3], photo, strongest) if st.local_colour else None
    if local is not None:
        flat[idx, :3] = np.clip(flat[idx, :3] + local, 0, 255)
    else:  # one gain and offset per channel, fitted where a view is fully trusted
        m = strongest >= 0.9
        if m.sum() >= 500:
            a, b = flat[idx, :3][m], photo[m]
            gain = np.clip(np.where(a.std(0) > 1e-6, b.std(0) / np.maximum(a.std(0), 1e-6), 1.0), 0.5, 2.0)
            flat[:, :3] = np.clip(flat[:, :3] * gain + np.clip(b.mean(0) - gain * a.mean(0), -64, 64), 0, 255)
    flat[idx, :3] = flat[idx, :3] * (1 - strongest[:, None]) + photo * strongest[:, None]
    wmap = np.zeros(h * w)
    wmap[idx] = strongest
    wmap = wmap.reshape(h, w)
    if local is not None:  # every texel changed colour, seen or not
        out, _ = grow_into_gutter(out, covered.astype(np.float64), covered, st.gutter_px)
        _, wmap = grow_into_gutter(out.copy(), wmap, covered, st.gutter_px)
    else:
        out, wmap = grow_into_gutter(out, wmap, covered, st.gutter_px)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8), wmap


# --- which views ---------------------------------------------------------------------------------------------------------

DEFAULT_VIEWS = ("front", "back")  # the angles whose outlines line up with the model (generated side and three-quarter
                                   # views drift a few pixels in pose, and paint features beside where the model has them)


def run_views(run, wanted=None):
    """{view: (image path, cache tag)} for a run, at the versions it used (RUN/views.json): the views asked for, else
    those of DEFAULT_VIEWS it was made from. A run made from one image (views.json "image") has that image as its only
    view, taken as the front: the outline fit decides whether it is one (a picture from another angle fits badly and
    is given no trust). Empty when the run has none of the views to paint from."""
    made = json.loads((run / "views.json").read_text()) if (run / "views.json").exists() else {}
    if made.get("image") and not made.get("set"):
        if wanted and list(wanted) != ["front"]:
            raise SystemExit("a run made from one image has only that image, taken as the front view")
        p = OUT / made["image"]
        if not p.exists():
            raise SystemExit(f"front: the run's image is gone ({p})")
        return {"front": (p, "image")}
    vdir = run.parent / "views"
    if not vdir.is_dir() and made.get("views_dir"):
        vdir = OUT / made["views_dir"] / "views"
    meta = json.loads((vdir / "views.json").read_text()) if (vdir / "views.json").exists() else {}
    versions = made.get("versions", {})
    names = wanted or [n for n in DEFAULT_VIEWS if n in versions] or ([] if made else ["front"])
    paths = {}
    for n in names:
        if n not in ANGLES:
            raise SystemExit(f"{n}: not a view this can place (one of {', '.join(ANGLES)})")
        ver, cur = versions.get(n), meta.get(n, {}).get("version")
        p = vdir / f"{n}.png" if ver is None or ver == cur else vdir / "history" / f"{n}.v{ver}.png"
        if not p.exists():
            raise SystemExit(f"{n}: no image at {p}")
        paths[n] = (p, f"v{ver or cur}")
    return paths


def views_run(folder):
    """The TRELLIS run whose views paint a model in `folder`: the folder itself, or for an optimised copy
    (rigging/optimise.py, optimise.json "from") the run it was made from. None if neither has a views.json."""
    folder = Path(folder)
    if (folder / "views.json").exists():
        return folder
    opt = folder / "optimise.json"
    if opt.exists():
        src = (folder.parent.parent / json.loads(opt.read_text())["from"]).parent
        if (src / "views.json").exists():
            return src
    return None


def paint_from_views(gpos, uvs, faces, texture, run, wanted=None, st=Settings()):
    """The base colour `texture` (H, W, 3 or 4) of a mesh in the GLB's frame (positions `gpos`, glTF `uvs`, triangles
    `faces`) painted from the views of TRELLIS run `run`. Returns (texture, trust map, view names), or None when the run
    has no views to paint from or none fits the model."""
    pos = trellis_frame(gpos)
    views = []
    for n, (p, tag) in run_views(run, wanted).items():
        rgba = cutout(p, run / "reproject" / f"{n}.{tag}_cutout.png")
        figure = np.ptp(np.nonzero(rgba[..., 3] > 127)[0])
        px = np.ptp(pos[:, 2]) / min(figure, 1536)  # the camera grid as fine as the picture
        views.append(place_view(n, rgba, pos, faces, uvs, texture, px, st))
    views = [v for v in views if v.trust > 0]  # a view whose outline doesn't fit the model's would paint nothing
    if not views:
        return None
    painted, wmap = paint(texture, gpos, uvs, faces, views, pos, st)
    return painted, wmap, [v.name for v in views]


def is_reprojected(model):
    """Whether a model is a reprojected one (named as main() names it): copies made of it from the run's voxels
    (rigging/optimise.py, rigging/item_game.py) are painted from the views too."""
    return Path(model).stem.endswith("_reprojected")


def paint_trimesh(mesh, run, st=Settings()):
    """paint_from_views for a trimesh with a PBR base colour texture, as o_voxel's to_glb and bake_textured make (the
    GLB's frame; trimesh keeps UVs with 0,0 at the bottom left), its texture replaced in place. Returns the share of the
    texture taken from the views, or None when the run has none that fits."""
    uv = np.asarray(mesh.visual.uv, np.float64).copy()
    uv[:, 1] = 1 - uv[:, 1]
    pic = mesh.visual.material.baseColorTexture
    mode = "RGBA" if "A" in pic.getbands() else "RGB"
    r = paint_from_views(np.asarray(mesh.vertices, np.float64), uv, np.asarray(mesh.faces, np.int64),
                         np.asarray(pic.convert(mode)), run, st=st)
    if r is None:
        return None
    mesh.visual.material.baseColorTexture = Image.fromarray(r[0], mode)
    return float((r[1] > 0.5).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=Path, help="a TRELLIS run's textured GLB")
    ap.add_argument("--views", help="comma-separated view names (default: the ones the run was made from)")
    ap.add_argument("--out", type=Path, help="default RUN/<model>_reprojected.glb")
    ap.add_argument("--weights", action="store_true", help="also write the trust map, reproject/<model>_trust.png beside the model")
    ap.add_argument("--no-nudge", action="store_true", help="don't move the views' features onto the model's")
    ap.add_argument("--no-edge-fade", action="store_true", help="no fade beside inner edges (it can darken deep eye "
                    "sockets and layered hair)")
    ap.add_argument("--no-local-colour", action="store_true", help="one colour shift for the unseen paint, not local")
    a = ap.parse_args()
    t0 = time.time()
    stem = a.model.stem
    run = views_run(a.model.parent) or a.model.parent  # an optimised copy: the views of the run it was made from
    st = Settings(nudge=not a.no_nudge, local_colour=not a.no_local_colour, inner_edge=0 if a.no_edge_fade else 8)
    gpos, uvs, faces, texture = read_glb(a.model)
    r = paint_from_views(gpos, uvs, faces, texture, run, a.views.split(",") if a.views else None, st)
    if r is None:
        print(f"{run.name}: nothing to paint from (made from none of {', '.join(DEFAULT_VIEWS)}, or no view's outline fits "
              "the model's: see the fits above): no copy written")
        return
    painted, wmap, names = r
    out = a.out or a.model.parent / f"{stem}_reprojected.glb"
    buf = io.BytesIO()
    Image.fromarray(painted).save(buf, format="PNG")
    out.write_bytes(replace_base_colour(a.model.read_bytes(), buf.getvalue()))
    if a.weights:
        (a.model.parent / "reproject").mkdir(exist_ok=True)
        Image.fromarray(np.rint(wmap * 255).astype(np.uint8)).save(a.model.parent / "reproject" / f"{stem}_trust.png")
    print(f"{len(names)} view(s), {(wmap > 0.5).mean():.0%} of the texture from them, {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
