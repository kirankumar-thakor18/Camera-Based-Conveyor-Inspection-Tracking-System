"""
Module C - Motion: how fast is the belt moving?

 (a) Lucas-Kanade written from scratch (brightness-constancy least squares per window,
     closed-form 2x2 solve, coarse-to-fine pyramid with warping) + arrow visualisation
 (b) OpenCV sparse feature-based flow (goodFeaturesToTrack + calcOpticalFlowPyrLK)
 (c) 6-parameter affine flow model fitted on one box with RANSAC (inliers green / outliers red)
 (d) belt speed in cm/s using the Module B calibration

Run: python module_C_optical_flow/optical_flow.py --video data/conveyor.mp4 [--frame 60] [--box-height-cm 5]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from common import OUT, ROOT, estimate_background, load_calibration, read_video, scale_K
from module_A_segmentation.segmentation import segment_boxes
from module_B_calibration.measure_box import Metric


# ============================================================ (a) Lucas-Kanade from scratch
def lk_dense(I1, I2, win=15, eig_thr=2.0):
    """
    Brightness constancy:  Ix*u + Iy*v + It = 0.   For every pixel stack the equation over a win x win
    window:  (A^T A) [u v]^T = -A^T It   with A^T A = [[sum Ix^2, sum IxIy],[sum IxIy, sum Iy^2]].
    Window sums are box filters; the 2x2 system is solved in closed form.  A pixel is 'valid' only if
    the smaller eigenvalue of A^T A is large enough (otherwise: aperture problem / no texture).
    """
    I1, I2 = I1.astype(np.float32), I2.astype(np.float32)
    Im = 0.5 * (I1 + I2)
    Ix = cv2.Sobel(Im, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    Iy = cv2.Sobel(Im, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    It = I2 - I1
    S = lambda x: cv2.boxFilter(x, -1, (win, win), normalize=False, borderType=cv2.BORDER_REPLICATE)
    Sxx, Sxy, Syy, Sxt, Syt = S(Ix * Ix), S(Ix * Iy), S(Iy * Iy), S(Ix * It), S(Iy * It)
    det = Sxx * Syy - Sxy * Sxy
    tr = Sxx + Syy
    lam_min = 0.5 * (tr - np.sqrt(np.maximum(tr * tr - 4 * det, 0)))
    valid = lam_min > eig_thr * win * win
    det_safe = np.where(valid, det, 1.0)
    u = (-Syy * Sxt + Sxy * Syt) / det_safe
    v = (Sxy * Sxt - Sxx * Syt) / det_safe
    u[~valid] = 0
    v[~valid] = 0
    return u, v, valid


def lk_pyramid(g1, g2, levels=4, win=15, eig_thr=2.0, iters=2):
    """Coarse-to-fine LK: estimate at the coarsest level, upsample, warp frame 2, refine the residual."""
    p1, p2 = [g1.astype(np.float32)], [g2.astype(np.float32)]
    for _ in range(levels - 1):
        p1.append(cv2.pyrDown(p1[-1]))
        p2.append(cv2.pyrDown(p2[-1]))
    flow = np.zeros(p1[-1].shape + (2,), np.float32)
    valid = None
    for lvl in range(levels - 1, -1, -1):
        h, w = p1[lvl].shape
        if flow.shape[:2] != (h, w):
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) * 2.0
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        for _ in range(iters):
            warped = cv2.remap(p2[lvl], xx + flow[..., 0], yy + flow[..., 1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            du, dv, valid = lk_dense(p1[lvl], warped, win, eig_thr)
            flow[..., 0] += du
            flow[..., 1] += dv
        flow[..., 0] = cv2.medianBlur(flow[..., 0], 5)  # light regularisation: remove isolated spikes
        flow[..., 1] = cv2.medianBlur(flow[..., 1], 5)
    return flow[..., 0], flow[..., 1], valid


def draw_flow(img, pts, vecs, color, scale=1.0, thickness=1):
    out = img.copy()
    for (x, y), (u, v) in zip(pts, vecs):
        cv2.arrowedLine(out, (int(x), int(y)), (int(x + scale * u), int(y + scale * v)), color, thickness, tipLength=0.35)
    return out


def dense_arrows(img, u, v, valid, step=14, scale=2.0):
    ys, xs = np.mgrid[step // 2:u.shape[0]:step, step // 2:u.shape[1]:step]
    ys, xs = ys.ravel(), xs.ravel()
    k = valid[ys, xs] & (np.hypot(u[ys, xs], v[ys, xs]) > 0.3)
    return draw_flow(img, np.c_[xs[k], ys[k]], np.c_[u[ys, xs][k], v[ys, xs][k]], (0, 0, 255), scale)


# ============================================================ (b) sparse feature-based flow
def sparse_flow(g1, g2, mask=None, max_corners=400):
    p0 = cv2.goodFeaturesToTrack(g1, max_corners, 0.01, 7, mask=mask, blockSize=7)
    if p0 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    return p0.reshape(-1, 2)[ok], (p1 - p0).reshape(-1, 2)[ok]


# ============================================================ (c) affine flow + RANSAC
def fit_affine(pts, flow):
    """u = a1 + a2 x + a3 y ; v = a4 + a5 x + a6 y  (least squares)."""
    A = np.c_[np.ones(len(pts)), pts[:, 0], pts[:, 1]]
    au, *_ = np.linalg.lstsq(A, flow[:, 0], rcond=None)
    av, *_ = np.linalg.lstsq(A, flow[:, 1], rcond=None)
    return np.r_[au, av]


def affine_predict(p, pts):
    A = np.c_[np.ones(len(pts)), pts[:, 0], pts[:, 1]]
    return np.c_[A @ p[:3], A @ p[3:]]


def _degenerate(pts):
    """True if the 3-point sample cannot pin down an affine (all points coincide or are collinear).

    np.linalg.lstsq happily returns a minimum-norm solution for a rank-deficient system, so an
    unguarded collinear sample produces a garbage hypothesis that only the consensus test weeds out.
    Testing the rank up front is cheaper and keeps the hypothesis count honest.
    """
    d = pts - pts.mean(axis=0)
    return abs(d[0, 0] * d[1, 1] - d[0, 1] * d[1, 0]) < 1e-9


def ransac_affine(pts, flow, iters=400, thr=0.6, seed=0):
    rng = np.random.default_rng(seed)
    n = len(pts)
    if n < 3:
        raise ValueError("need >= 3 flow vectors")
    best = None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if _degenerate(pts[idx]):
            continue
        p = fit_affine(pts[idx], flow[idx])
        inl = np.linalg.norm(affine_predict(p, pts) - flow, axis=1) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        # Every sampled hypothesis was degenerate - fall back on a plain least-squares fit so the
        # caller still gets a number, and let the caller's inlier-count check reject it.
        p = fit_affine(pts, flow)
        return p, np.linalg.norm(affine_predict(p, pts) - flow, axis=1) < thr
    p = fit_affine(pts[best], flow[best])  # refit on the consensus set
    for _ in range(2):  # refit on the refined inlier set, re-deriving inliers each pass so that the
        inl = np.linalg.norm(affine_predict(p, pts) - flow, axis=1) < thr  # returned model and the
        if inl.sum() < 3:  # returned inlier mask always describe the SAME fit
            break
        p = fit_affine(pts[inl], flow[inl])
    inl = np.linalg.norm(affine_predict(p, pts) - flow, axis=1) < thr
    return p, inl


def box_flow_speed(g1, g2, mask, fps, metric, u=None, v=None, valid=None, want_debug=False):
    """Affine+RANSAC flow of one box -> translation at its centroid -> cm/s.  Used by pipeline.py."""
    if u is None:
        u, v, valid = lk_pyramid(g1, g2)
    er = cv2.erode(mask, np.ones((7, 7), np.uint8))
    ys, xs = np.where((er > 0) & valid)
    sel = slice(None, None, max(1, len(xs) // 1500))
    pts = np.c_[xs, ys][sel].astype(np.float64)
    fl = np.c_[u[ys, xs], v[ys, xs]][sel].astype(np.float64)
    if len(pts) < 12:  # textureless -> fall back on sparse corners inside the mask
        sp, sf = sparse_flow(g1, g2, mask=er)
        pts, fl = sp.astype(np.float64), sf.astype(np.float64)
    if len(pts) < 3:
        return None
    par, inl = ransac_affine(pts, fl)
    M = cv2.moments(mask, binaryImage=True)
    c = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
    d = affine_predict(par, c[None])[0]  # px/frame at the centroid
    cm_per_frame = metric.displacement_cm(c, c + d)
    res = dict(px_per_frame=d, speed_cm_s=cm_per_frame * fps, n_inliers=int(inl.sum()), n_total=len(pts), params=par)
    if want_debug:
        res.update(pts=pts, flow=fl, inliers=inl)
    return res


# ============================================================ demo
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--calib", default=str(ROOT / "module_B_calibration" / "calibration.npz"))
    ap.add_argument("--box-height-cm", type=float, default=None, help="height of box TOP face above the belt (required for a cm/s estimate)")
    ap.add_argument("--px-per-cm", type=float, default=None, help="fallback if no calibration")
    ap.add_argument("--box-index", type=int, default=0, help="which detected box (largest first)")
    ap.add_argument("--no-undistort", action="store_true", help="skip Module B undistortion (not recommended)")
    a = ap.parse_args()

    frames, fps, s = read_video(a.video, resize_width=640)
    bg = estimate_background(frames)
    i = a.frame if a.frame is not None else len(frames) // 2
    if not 0 <= i < len(frames) - 1:
        sys.exit(f"--frame {i} out of range: the video has {len(frames)} frames, so a consecutive pair needs 0..{len(frames) - 2}")
    f1, f2 = frames[i], frames[i + 1]
    g1, g2 = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f1, f2))

    cal = load_calibration(a.calib)
    if cal is not None and a.px_per_cm is None:
        K, dist = scale_K(cal["K"], s), cal["dist"]
        metric = Metric(K, cal["belt_R"], cal["belt_t"], a.box_height_cm or 0.0)
        # The pixel displacements below are measured in whatever frame the images are in, so the
        # calibration must describe THAT same frame.  pipeline.py undistorts, so we do too -
        # otherwise this module and the pipeline quote systematically different cm/s for one video.
        if not a.no_undistort:
            frames = [cv2.undistort(f, K, dist) for f in frames]
            bg = estimate_background(frames)
            f1, f2 = frames[i], frames[i + 1]
            g1, g2 = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (f1, f2))
            print(f"[B] calibration loaded (f={K[0,0]:.0f}px) + undistortion applied")
    else:
        if a.px_per_cm is None:
            sys.exit("No calibration found: run Module B or pass --px-per-cm")
        metric = Metric(px_per_cm=a.px_per_cm)

    # The px->cm scale depends on the assumed height of the box top face: a wrong height rescales the
    # belt plane and therefore rescales cm/s directly (measured: h=0 -> +9%, h=2cm -> +5%, h=8cm -> -6%).
    if metric.calibrated and not a.box_height_cm:
        print("[B] WARNING --box-height-cm not given (assuming 0).  The speed will be biased; "
              "pass --box-height-cm <top-face height in cm> for a usable number.")

    # (a) hand-built dense LK
    u, v, valid = lk_pyramid(g1, g2)
    cv2.imwrite(str(OUT / "C_a_lucas_kanade_dense.png"), dense_arrows(f1, u, v, valid))
    # (b) sparse feature flow
    sp, sf = sparse_flow(g1, g2)
    cv2.imwrite(str(OUT / "C_b_sparse_lk.png"), draw_flow(f1, sp, sf, (0, 200, 0), 2.0, 1))
    # comparison numbers -----------------------------------------------------------------
    xi, yi = sp[:, 0].astype(int), sp[:, 1].astype(int)
    both = valid[yi, xi]
    dif = np.hypot(u[yi, xi] - sf[:, 0], v[yi, xi] - sf[:, 1])
    print(f"[compare] sparse corners: {len(sp)} | of them with valid dense-LK estimate: {both.sum()} | "
          f"median |dense - sparse| on those = {np.median(dif[both]) if both.any() else float('nan'):.2f} px")
    print(f"[compare] pixels where dense LK is invalid (aperture problem / no texture): {100*(~valid).mean():.1f}%")
    # COMMENT (dense vs sparse): where they agree - well-textured corners on box labels/edges. Where they
    # disagree - (1) textureless box faces: A^T A is (nearly) singular so dense LK is masked out or,
    # along a straight edge, only the flow component NORMAL to the edge is observable (aperture problem);
    # sparse corner tracking never even tries there. (2) large motions: single-scale differential LK
    # violates the small-motion assumption, hence the pyramid; sparse LK also uses a pyramid.

    # (c) affine + RANSAC on one box
    dets = sorted(segment_boxes(f1, bg), key=lambda d: -d["area"])
    if not dets:
        sys.exit("No box segmented in this frame - try another --frame")
    box = dets[min(a.box_index, len(dets) - 1)]
    res = box_flow_speed(g1, g2, box["mask"], fps, metric, u, v, valid, want_debug=True)
    if res is None:
        sys.exit("Not enough flow vectors on the box.")
    if res["n_inliers"] < 8:
        print(f"[affine] WARNING only {res['n_inliers']}/{res['n_total']} vectors agreed with the affine "
              f"model - the fit is not trustworthy, so treat the cm/s below as meaningless. "
              f"Try another --frame or a box with more texture.")
    vis = f1.copy()
    cv2.drawContours(vis, [box["contour"]], -1, (255, 255, 0), 1)
    for (x, y), (du, dv), ok in zip(res["pts"], res["flow"], res["inliers"]):
        cv2.arrowedLine(vis, (int(x), int(y)), (int(x + 2 * du), int(y + 2 * dv)),
                        (0, 220, 0) if ok else (0, 0, 255), 1, tipLength=0.4)
    cv2.imwrite(str(OUT / "C_c_affine_ransac.png"), vis)
    print("[affine] params [a1 a2 a3 | a4 a5 a6] =", np.round(res["params"], 4))
    print(f"[ransac] inliers {res['n_inliers']}/{res['n_total']} (green = inlier, red = outlier)")

    # (d) speed
    print(f"[speed] box translation = ({res['px_per_frame'][0]:.2f}, {res['px_per_frame'][1]:.2f}) px/frame at {fps:.1f} fps")
    if metric.calibrated:
        how = f"calibrated, box-top plane h={a.box_height_cm:.1f} cm" if a.box_height_cm else "calibrated but h UNKNOWN - see warning above"
    else:
        how = "px_per_cm fallback"
    print(f"[speed] BELT SPEED ESTIMATE = {res['speed_cm_s']:.2f} cm/s  ({how})")
    print("Saved images to", OUT)


if __name__ == "__main__":
    main()
