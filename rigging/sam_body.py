"""SAM 3D Body on a clip's frames or on still renders: per frame, the Momentum Human Rig's joint rotations and positions.

python -m rigging.sam_body OUT.npz VIDEO.mp4 [--calib DET.npz] [--view x0,x1]
python -m rigging.sam_body OUT.npz IMG.png [IMG.png ...]

OUT.npz:
  rots  (T, 127, 3, 3)  each MHR joint's world rotation in camera axes (x right, y up, z toward the camera)
  joints (T, 127, 3)    each MHR joint's position in the same axes, metres, plus the camera offset (cam_t)
  kp2d  (T, 70, 2)      SAM's 70 keypoints in the image's pixels (MHR70 order)
  names (127,)          MHR joint names; parents (127,) their parent indices; kp_names (70,) the keypoints' names
A video is read at layout size (1280 x 720, through DET.npz's calibration when given) and only the front view is used
(--view, default the first 420 px): that is the two-view clips' front half. Stills are used whole.

The body sits on a plain studio backdrop, so its box is the region that differs from the backdrop colour.
The camera is given a long lens (FOCAL x the image height): the clips' front view is orthographic.

Needs a Python with sam_3d_body importable (SAM3D_BODY_DIR, default ~/src/sam-3d-body) and access to the gated
weights on Hugging Face (SAM3D_BODY_REPO, default facebook/sam-3d-body-dinov3). SAM License, commercial use allowed.
"""
import contextlib, io, os, sys

import cv2
import numpy as np
import torch

FOCAL = 4.0
H, W = 720, 1280


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")


def load_estimator(dev):
    sys.path.insert(0, os.path.expanduser(os.environ.get("SAM3D_BODY_DIR", "~/src/sam-3d-body")))
    with contextlib.redirect_stderr(io.StringIO()):
        import sam_3d_body.sam_3d_body_estimator as est
        from sam_3d_body.build_models import _hf_download, load_sam_3d_body
    if dev.type != "cuda":  # the estimator moves its batches to "cuda" by name, from several modules
        to = est.recursive_to
        moved = lambda x, t: to(x, dev if t == "cuda" else t)
        for m in [m for k, m in sys.modules.items() if k.startswith("sam_3d_body") and hasattr(m, "recursive_to")]:
            m.recursive_to = moved
        torch.cuda.empty_cache = lambda: None
        torch.Tensor.cuda = lambda self, *a, **k: self.to(dev)
    ckpt, mhr = _hf_download(os.environ.get("SAM3D_BODY_REPO", "facebook/sam-3d-body-dinov3"))
    model, cfg = load_sam_3d_body(ckpt, device=dev, mhr_path=mhr)
    return est.SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)


def skeleton(mhr_path=None):
    """MHR's joint names and parents, from its TorchScript model."""
    if mhr_path is None:
        sys.path.insert(0, os.path.expanduser(os.environ.get("SAM3D_BODY_DIR", "~/src/sam-3d-body")))
        from sam_3d_body.build_models import _hf_download
        _, mhr_path = _hf_download(os.environ.get("SAM3D_BODY_REPO", "facebook/sam-3d-body-dinov3"))
    sk = torch.jit.load(mhr_path, map_location="cpu").character_torch.skeleton
    return list(sk.joint_names), sk.joint_parents.numpy()


def body_box(img, pad=0.08):
    """The box around whatever differs from the backdrop (its colour taken from the image's top corners)."""
    bg = np.median(np.concatenate([img[:20, :20].reshape(-1, 3), img[:20, -20:].reshape(-1, 3)]), 0)
    fg = np.abs(img.astype(np.int16) - bg).sum(-1) > 30
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(fg)
    if len(xs) < 100:
        return np.array([0, 0, img.shape[1], img.shape[0]], np.float32)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    g = pad * max(x1 - x0, y1 - y0)
    return np.array([max(0, x0 - g), max(0, y0 - g), min(img.shape[1], x1 + g), min(img.shape[0], y1 + g)], np.float32)


def read(estimator, images):
    rots, joints, kp2d = [], [], []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        f = FOCAL * h
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            out = estimator.process_one_image(np.ascontiguousarray(img), bboxes=body_box(img)[None], cam_int=torch.from_numpy(K[None]),
                                              inference_type="full")
        o = out[0]
        # joint positions come out in OpenCV camera axes (y down, z away); rotations in MHR's own (y up, z toward)
        j = (o["pred_joint_coords"] + o["pred_cam_t"]) * np.array([1, -1, -1])
        rots.append(o["pred_global_rots"])
        joints.append(j)
        kp2d.append(o["pred_keypoints_2d"][:70, :2])
        print(f"\r  SAM 3D Body {i + 1}/{len(images)}", end="", flush=True)
    print()
    return np.array(rots), np.array(joints), np.array(kp2d)


def video_views(path, calib=None, view=(0, 420)):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.warpAffine(f, calib, (W, H), flags=cv2.INTER_AREA) if calib is not None else cv2.resize(f, (W, H))
        out.append(f[:, view[0]:view[1], ::-1])
    return out, fps


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opt = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
    out, src = args[0], args[1:]
    dev = device()
    print("SAM 3D Body on", dev)
    estimator = load_estimator(dev)
    if src[0].endswith((".mp4", ".mov", ".webm", ".mkv")):
        calib = np.load(opt("--calib"))["calib"] if opt("--calib") else None
        view = tuple(int(x) for x in opt("--view").split(",")) if opt("--view") else (0, 420)
        images, fps = video_views(src[0], calib, view)
    else:
        images, fps = [cv2.imread(p)[..., ::-1] for p in src], 0.0
    rots, joints, kp2d = read(estimator, images)
    names, parents = skeleton()
    from sam_3d_body.metadata.mhr70 import mhr_names
    np.savez(out, rots=rots, joints=joints, kp2d=kp2d, names=np.array(names), parents=parents, fps=fps,
             kp_names=np.array(mhr_names[:70]),
             size=np.array([images[0].shape[1], images[0].shape[0]]), focal=FOCAL)
    print("saved", out)


if __name__ == "__main__":
    main()
