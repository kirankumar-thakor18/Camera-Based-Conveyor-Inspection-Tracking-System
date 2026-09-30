"""
Module D - Kalman filter tracking (written from scratch).

State x = [x, y, vx, vy]^T, constant-velocity model, measurement z = [x, y] (centroid from Module A).
  predict :  x- = F x            P- = F P F^T + Q
  update  :  y = z - H x-        S = H P- H^T + R      K = P- H^T S^-1
             x = x- + K y        P = (I-KH) P- (I-KH)^T + K R K^T   (Joseph form)
Also contains MultiBoxTracker (greedy/Hungarian association + one Kalman filter per box) used by pipeline.py.

Run: python module_D_tracking/kalman_tracker.py --video data/conveyor.mp4 [--dropout-start 40 --dropout-len 5]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from common import OUT, estimate_background, read_video
from module_A_segmentation.segmentation import segment_boxes


class KalmanFilter:
    def __init__(self, z0, dt=1.0, sigma_a=0.3, sigma_z=2.0):
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        G = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])  # acceleration noise
        self.Q = G @ G.T * sigma_a ** 2
        self.R = np.eye(2) * sigma_z ** 2
        self.x = np.array([z0[0], z0[1], 0.0, 0.0])
        self.P = np.diag([sigma_z ** 2, sigma_z ** 2, 100.0, 100.0])  # velocity unknown at start

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, z):
        y = np.asarray(z, float) - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x[:2].copy()


def observability_rank(kf):
    O = np.vstack([kf.H @ np.linalg.matrix_power(kf.F, k) for k in range(4)])
    return np.linalg.matrix_rank(O)


# =============================================================== multi-box tracker (pipeline)
class Track:
    def __init__(self, tid, det, sigma_z):
        self.id = tid
        self.kf = KalmanFilter(det["centroid"], sigma_z=sigma_z)
        self.det = det
        self.hits, self.missed = 1, 0
        self.pred = np.array(det["centroid"])


class MultiBoxTracker:
    def __init__(self, gate=60.0, max_missed=8, sigma_z=2.0, min_hits=3):
        self.tracks, self.gate, self.max_missed, self.sigma_z, self.min_hits = [], gate, max_missed, sigma_z, min_hits
        self._next_id = 1  # per-instance, so a second tracker in the same process starts at #1 again

    def step(self, dets):
        for t in self.tracks:
            t.pred = t.kf.predict()
        matched_t, matched_d = set(), set()
        if self.tracks and dets:
            C = np.array([[np.linalg.norm(t.pred - np.array(d["centroid"])) for d in dets] for t in self.tracks])
            # Gate BEFORE solving.  linear_sum_assignment minimises the cost over the whole matrix, so if
            # we let it see raw distances a single hopeless pair (a far-away detection) can consume a
            # track and force a spurious birth.  Pushing inadmissible costs out of the way first makes
            # the solver prefer admissible pairs and only falls back to a gated pair if it must.
            BIG = self.gate * 1e3
            rows, cols = linear_sum_assignment(np.where(C < self.gate, C, BIG))
            for i, j in zip(rows, cols):
                if C[i, j] >= self.gate:
                    continue
                t = self.tracks[i]
                t.kf.update(dets[j]["centroid"])
                t.det, t.hits, t.missed = dets[j], t.hits + 1, 0
                matched_t.add(i)
                matched_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.missed += 1
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.tracks.append(Track(self._next_id, d, self.sigma_z))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.missed == 0]


# =============================================================== single-box demo with occlusion
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--dropout-start", type=int, default=40, help="frames after tracking starts")
    ap.add_argument("--dropout-len", type=int, default=5)
    ap.add_argument("--extra-noise", type=float, default=3.0,
                    help="std (px) of extra Gaussian noise added to the segmentation centroid (0 = none)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    frames, fps, _ = read_video(a.video, resize_width=640)
    bg = estimate_background(frames)
    H, W = frames[0].shape[:2]
    all_dets = [segment_boxes(f, bg) for f in frames]

    # target = largest fully-visible box in the first frame that has one
    start = next((i for i, d in enumerate(all_dets) if d), None)
    if start is None:
        sys.exit("No boxes segmented in the video.")
    tgt = max(all_dets[start], key=lambda d: d["area"])
    sigma_z = float(np.hypot(a.extra_noise, 1.5))
    kf = KalmanFilter(tgt["centroid"], sigma_z=sigma_z)
    print(f"Tracking starts at frame {start}, box centroid {np.round(tgt['centroid'], 1)}, "
          f"measurement std = {sigma_z:.2f} px")
    print(f"Observability matrix rank = {observability_rank(kf)} (state dim 4) -> position-only measurements "
          f"are enough to recover velocity")

    d0, d1 = start + a.dropout_start, start + a.dropout_start + a.dropout_len
    rows, raw, est = [], [], []
    for n in range(start + 1, len(frames)):
        pred = kf.predict()
        # 1-sigma spread of the predicted position.  During the dropout no update ever runs, so Q keeps
        # inflating P and this band widens - that growth IS the observability story the module is meant to show.
        sig = np.sqrt(np.maximum(np.diag(kf.P)[:2], 0.0))
        dets = all_dets[n]
        z = None
        if dets and not (d0 <= n < d1):  # simulated occlusion: discard measurement completely
            j = int(np.argmin([np.linalg.norm(np.array(d["centroid"]) - pred) for d in dets]))
            if np.linalg.norm(np.array(dets[j]["centroid"]) - pred) < 60:
                z = np.array(dets[j]["centroid"]) + rng.normal(0, a.extra_noise, 2)
        corr = kf.update(z) if z is not None else pred.copy()
        rows.append((n, pred.copy(), z, corr.copy(), d0 <= n < d1, sig))
        if kf.x[0] > W + 30 or kf.x[1] < -30 or kf.x[1] > H + 30 or kf.x[0] < -30:
            break

    print("\nframe |   predicted (x,y)  |  measurement (x,y)  |  corrected (x,y)   | note")
    show = [r for r in rows if r[0] < start + 13 or d0 - 2 <= r[0] < d1 + 3]
    for n, p, z, c, occ, sg in show:
        zs = "     --- MISSING ---   " if z is None else f"({z[0]:8.2f},{z[1]:8.2f})"
        print(f"{n:5d} | ({p[0]:7.2f},{p[1]:7.2f}) | {zs} | ({c[0]:7.2f},{c[1]:7.2f}) | "
              f"{'+-%.1fpx' % sg.mean():>9} {'OCCLUDED: prediction only' if occ else ''}")

    # ---------- plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = np.array([r[0] for r in rows])
    zx = np.array([np.nan if r[2] is None else r[2][0] for r in rows])
    zy = np.array([np.nan if r[2] is None else r[2][1] for r in rows])
    cx = np.array([r[3][0] for r in rows])
    cy = np.array([r[3][1] for r in rows])
    sx = np.array([r[5][0] for r in rows])
    sy = np.array([r[5][1] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for k, (zz, cc, ss, lab) in enumerate([(zx, cx, sx, "x [px]"), (zy, cy, sy, "y [px]")]):
        ax[k].plot(fr, zz, "r.", ms=4, label="raw measurement")
        ax[k].plot(fr, cc, "g-", lw=1.8, label="Kalman estimate")
        ax[k].fill_between(fr, cc - 2 * ss, cc + 2 * ss, color="green", alpha=0.15, label="Kalman +-2 sigma")
        ax[k].axvspan(d0, d1 - 1, color="orange", alpha=0.3, label="dropout (occlusion)")
        ax[k].set_xlabel("frame"); ax[k].set_ylabel(lab); ax[k].legend(fontsize=8)
    ax[2].plot(zx, zy, "r.", ms=4, label="raw")
    ax[2].plot(cx, cy, "g-", label="Kalman")
    m = np.array([r[4] for r in rows])
    ax[2].plot(cx[m], cy[m], "o", color="orange", label="predicted only")
    ax[2].invert_yaxis(); ax[2].set_xlabel("x"); ax[2].set_ylabel("y"); ax[2].legend(fontsize=8)
    ax[2].set_title("image-plane trajectory")
    fig.suptitle("Covariance grows while measurements are missing: the filter keeps predicting, "
                 "but its confidence drops - that is the observability argument", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "D_kalman_plot.png", dpi=120)

    # ---------- overlay video
    vw = cv2.VideoWriter(str(OUT / "D_kalman_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    trail = []
    for n, p, z, c, occ, sg in rows:
        im = frames[n].copy()
        trail.append(tuple(c.astype(int)))
        for q in range(1, len(trail)):
            cv2.line(im, trail[q - 1], trail[q], (0, 255, 0), 2)
        if z is not None:
            cv2.circle(im, tuple(z.astype(int)), 4, (0, 0, 255), -1)
        cv2.circle(im, tuple(p.astype(int)), 7, (255, 128, 0), 1)
        cv2.circle(im, tuple(c.astype(int)), 3, (0, 255, 0), -1)
        cv2.putText(im, "red=raw  green=Kalman  blue ring=prediction", (10, 20), 0, 0.5, (255, 255, 255), 1)
        if occ:
            cv2.putText(im, "OCCLUDED: prediction only", (10, 45), 0, 0.7, (0, 165, 255), 2)
        vw.write(im)
    vw.release()
    err = np.nanmean(np.hypot(zx - cx, zy - cy))
    print(f"\nmean |raw - Kalman| = {err:.2f} px | saved {OUT/'D_kalman_plot.png'} and {OUT/'D_kalman_overlay.mp4'}")


if __name__ == "__main__":
    main()
