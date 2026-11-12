"""Where SAM 3D Body puts each of its joints on the character at rest, in the world: triangulated from the rest renders.

python -m rigging.fit REST_DIR OUT.json

REST_DIR: render_rest.py's orthographic renders and rest_cams.json, with SAM's reading of them in rest_sam.npz.
Each view puts a joint on a line through the scene (orthographic: along the camera's axis, through where SAM projects
it in that image); the joint is the point nearest all the lines. A view more than OUTLIER metres off the others' point
(SAM swapping a turned figure's arms) is dropped. OUT.json: {joint: [x, y, z]} in metres, plus each one's spread.
"""
import json, sys

import numpy as np

OUTLIER = 0.05


def project(j, size, focal):
    """SAM's camera-axes joints (x right, y up, z toward the camera) to the image's pixels, through its long lens."""
    w, h = size
    f = focal * h
    z = -j[..., 2]
    return np.stack([w / 2 + f * j[..., 0] / z, h / 2 - f * j[..., 1] / z], -1)


def lines(uv, cam):
    """Pixels in an orthographic render to world lines: a point on each and the shared direction."""
    R = np.array(cam["rot"])
    w, h = cam["res"]
    px = cam["ortho_scale"] / h
    local = np.stack([(uv[..., 0] - w / 2) * px, (h / 2 - uv[..., 1]) * px, np.zeros(uv.shape[:-1])], -1)
    return np.array(cam["centre"]) + local @ R.T, R[:, 2]


def nearest(points, dirs):
    """The point nearest a set of lines (least squares)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for p, d in zip(points, dirs):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ p
    return np.linalg.lstsq(A, b, rcond=None)[0]


def fit(rest_dir):
    cams = json.load(open(f"{rest_dir}/rest_cams.json"))["cams"]
    r = np.load(f"{rest_dir}/rest_sam.npz")
    names = [str(n) for n in r["names"]]
    size, focal = r["size"], float(r["focal"])
    P, D = [], []
    for v, cam in enumerate(cams):
        p, d = lines(project(r["joints"][v], size, focal), cam)
        P.append(p)
        D.append(np.repeat(d[None], len(names), 0))
    P, D = np.array(P), np.array(D)
    out, spread = {}, {}
    for j, n in enumerate(names):
        keep = np.ones(len(cams), bool)
        for _ in range(len(cams) - 2):
            x = nearest(P[keep, j], D[keep, j])
            off = np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - p)) for p, d in zip(P[:, j], D[:, j])])
            worst = np.argmax(np.where(keep, off, -1))
            if off[worst] < OUTLIER or keep.sum() <= 2:
                break
            keep[worst] = False
        out[n] = x.round(4).tolist()
        spread[n] = round(float(off[keep].max()), 4)
    return out, spread


def main():
    rest_dir, out = sys.argv[1:3]
    joints, spread = fit(rest_dir)
    json.dump({"joints": joints, "spread_m": spread}, open(out, "w"), indent=1)
    worst = sorted(spread.items(), key=lambda kv: -kv[1])[:5]
    print(f"fitted {len(joints)} joints -> {out}; worst spread (m): {worst}")


if __name__ == "__main__":
    main()
