"""Shared helpers used by every module (video I/O, background model, calibration loading)."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def read_video(path, max_frames=None, step=1, resize_width=None):
    """Return (frames, effective_fps, scale). scale = resize factor applied to the frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, i, scale = [], 0, 1.0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % step == 0:
            if resize_width and f.shape[1] != resize_width:
                scale = resize_width / f.shape[1]
                # INTER_AREA is a DOWNsampling kernel.  Applying it to an upscale blurs the frame, so
                # pick the kernel from the direction of the resize.
                interp = cv2.INTER_AREA if f.shape[1] > resize_width else cv2.INTER_LINEAR
                f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=interp)
            frames.append(f)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames, fps / step, scale


def estimate_background(frames, n=40):
    """Median of evenly spaced frames -> static-camera background (boxes move, so they vanish)."""
    if not frames:
        raise ValueError("estimate_background: empty frame list")
    idx = np.linspace(0, len(frames) - 1, min(n, len(frames))).astype(int)
    return np.median(np.stack([frames[i] for i in idx]), axis=0).astype(np.uint8)


def load_calibration(path):
    """Load the .npz written by module_B_calibration/calibrate.py. Returns dict or None."""
    path = Path(path)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    return out


def scale_K(K, s):
    """Rescale an intrinsic matrix to a resized image (row scaling handles the skew term correctly)."""
    K2 = K.copy().astype(float)
    K2[0, :] *= s
    K2[1, :] *= s
    return K2
