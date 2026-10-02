"""
Render a self-consistent checkerboard photo set, so Module B's calibration has something to run on.

    python tools/make_demo_checkerboard.py

Why this exists
---------------
Module B needs >= 10 photos of a 9x6 board. Where there is no printer and no board has been
photographed, `cv2.calibrateCamera` has nothing to run on. Rather than hand-typing an intrinsic
matrix, this renders a *geometrically self-consistent* session: a known camera (K + distortion)
observes the board at 15 known poses. `calibrate.py` then recovers K, the distortion and the
belt-plane pose from those images exactly as it would from real photos - the optics being solved are
simulated, but the solve itself, the reprojection error and the P = K[R|t] decomposition are all
genuine output of the real code path.

Resolution (1920x1080) and focal length are chosen to match `data/conveyor.mp4`, so the demo is
dimensionally consistent with the footage the rest of the project uses.

Output goes to `data/checkerboard_demo/`, deliberately NOT `data/checkerboard/`, and the calibration
is written as `calibration_demo.npz`, deliberately NOT `calibration.npz`. That second part matters:
`optical_flow.py` and `pipeline.py` auto-load `calibration.npz` when it exists and would then report
centimetres derived from a simulated camera. With a demo-only filename they keep using the
known-size reference object for scale, which is the honest number for a clip shot by another camera.

To replace this with a real calibration, shoot the printed board and run:

    python tools/shoot_checkerboard.py --camera 0
    python module_B_calibration/calibrate.py --images data/checkerboard --pattern 9x6 \
           --square-mm 25 --belt-image belt_00.png
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "checkerboard_demo"

W, H = 1920, 1080            # matches data/conveyor.mp4
COLS, ROWS = 9, 6            # inner corners
SQ = 2.5                     # cm per square
BW_PX, BH_PX = 1200, 840     # board raster
S = (COLS + 1) * SQ / BW_PX  # cm per board pixel

# plausible laptop-webcam intrinsics: ~58 deg horizontal FOV at 1920 px wide
F = 1560.0
KK = np.array([[F, 0.0, W / 2.0],
               [0.0, F * 1.02, H / 2.0],
               [0.0, 0.0, 1.0]])
DIST = np.array([[-0.21, 0.06, 0.0004, -0.0006, 0.0]])


def make_board():
    board = np.zeros((BH_PX, BW_PX), np.uint8)
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                board[r * BH_PX // (ROWS + 1):(r + 1) * BH_PX // (ROWS + 1),
                      c * BW_PX // (COLS + 1):(c + 1) * BW_PX // (COLS + 1)] = 255
    return board


BOARD = make_board()


def render(R, t):
    """Draw the board on a grey backdrop under pose (R, t), then apply barrel distortion."""
    P = KK @ np.c_[R, t]
    # board raster px -> board cm is  x = S*(px - BW_PX/2), y = S*(py - BH_PX/2), z = 0
    M = np.array([
        [P[0, 0] * S, P[0, 1] * S, P[0, 3] - S * (P[0, 0] * BW_PX / 2 + P[0, 1] * BH_PX / 2)],
        [P[1, 0] * S, P[1, 1] * S, P[1, 3] - S * (P[1, 0] * BW_PX / 2 + P[1, 1] * BH_PX / 2)],
        [P[2, 0] * S, P[2, 1] * S, P[2, 3] - S * (P[2, 0] * BW_PX / 2 + P[2, 1] * BH_PX / 2)],
    ])
    mask = cv2.warpPerspective(np.full_like(BOARD, 255), M, (W, H), borderValue=0)
    img = cv2.warpPerspective(BOARD, M, (W, H), borderValue=0)
    out = np.where(mask > 0, img, 120).astype(np.uint8)

    # for each output pixel, find its source in the undistorted image
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
    und = cv2.undistortPoints(pix.reshape(-1, 1, 2), KK, DIST).reshape(-1, 2)
    mx = (und[:, 0] * KK[0, 0] + KK[0, 2]).astype(np.float32).reshape(H, W)
    my = (und[:, 1] * KK[1, 1] + KK[1, 2]).astype(np.float32).reshape(H, W)
    out = cv2.remap(out, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return cv2.GaussianBlur(out, (3, 3), 0.6)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    names = []

    # shot 0: flat and square-on -- becomes belt_00.png, the belt-plane view
    R0, _ = cv2.Rodrigues(np.array([0.06, -0.05, 0.015], np.float64))
    cv2.imwrite(str(OUT / "belt_00.png"), render(R0, np.array([[0.0], [0.0], [-115.0]])))
    names.append("belt_00.png")

    # shots 1-2: still flat, different distances and off-centre placement
    for i, (tx, ty, tz) in enumerate([(-22.0, -12.0, -100.0), (26.0, 14.0, -138.0)]):
        R, _ = cv2.Rodrigues(np.array([0.05, -0.04, 0.01 * i], np.float64))
        nm = f"flat_{i + 1:02d}.png"
        cv2.imwrite(str(OUT / nm), render(R, np.array([[tx], [ty], [tz]])))
        names.append(nm)

    # shots 3-14: tilted and rolled views at varied distance. A single frontal plane of views cannot
    # separate k1 from k2 from k3, because all three are radial -- tilt is what pins them down.
    for i in range(12):
        rx, ry, rz = rng.uniform(-0.48, 0.48), rng.uniform(-0.48, 0.48), rng.uniform(-0.55, 0.55)
        R, _ = cv2.Rodrigues(np.array([rx, ry, rz], np.float64))
        t = np.array([[rng.uniform(-30.0, 30.0)],
                      [rng.uniform(-20.0, 20.0)],
                      [-rng.uniform(95.0, 190.0)]])
        nm = f"tilt_{i + 1:02d}.png"
        cv2.imwrite(str(OUT / nm), render(R, t))
        names.append(nm)

    print(f"wrote {len(names)} views to {OUT}")
    print(f"ground-truth camera: fx={KK[0, 0]:.0f} fy={KK[1, 1]:.0f} "
          f"cx={KK[0, 2]:.0f} cy={KK[1, 2]:.0f}")
    print(f"ground-truth distortion: {DIST.ravel()}")
    print("\nnow solve it with the real code:")
    print(f'  python module_B_calibration/calibrate.py --images data/checkerboard_demo '
          f'--pattern 9x6 --square-mm 25 \\\n      --belt-image belt_00.png '
          f'--out module_B_calibration/calibration_demo.npz')
    return 0


if __name__ == "__main__":
    sys.exit(main())