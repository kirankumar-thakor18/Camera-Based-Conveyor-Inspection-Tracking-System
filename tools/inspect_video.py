"""Validate a self-recorded conveyor clip against what the pipeline actually needs.

The assignment's conclusions depend on properties of the footage that are easy to
miss while recording and impossible to fix afterwards. This reports them up front.

    python tools/inspect_video.py data/conveyor.mp4

Read-only: it does not write into the project.
"""
import argparse
import os
import sys

import cv2
import numpy as np

HEVC_FOURCC = cv2.VideoWriter_fourcc(*"hevc")
H264_FOURCC = cv2.VideoWriter_fourcc(*"avc1")


def open_capture(path):
    """Return (capture, note). Mobile clips are frequently HEVC in .MOV, which
    OpenCV often cannot decode - that is a re-export problem, not a code problem."""
    if not os.path.exists(path):
        sys.exit(f"  ERROR: no such file: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(
            f"  ERROR: OpenCV cannot decode {path}\n"
            "         Most likely cause: HEVC/H.265 in a .MOV container (iPhone default).\n"
            "         Re-export as H.264 MP4 and retry."
        )
    return cap, ""


def static_mask(grays, tol=25):
    """Boolean mask of pixels that never change across the sampled frames.

    A conveyor belt is usually textured, and that texture scrolls past a bolted-down camera. ORB
    therefore finds plenty of high-quality keypoints that legitimately move, and a homography fitted
    to them reports tens of pixels of "camera drift" for footage where the camera never budges. On
    data/conveyor.mp4 that mistake reads 49.95 px where the true drift is under 0.1 px.

    A pixel counts as static only if it stays within `tol` grey levels of the median image across
    EVERY sampled frame. A belt pixel fails this as soon as its texture has scrolled away; a wall,
    frame rail or machine housing passes at every sample. Returns (mask, median_image).
    """
    stack = np.stack(grays).astype(np.int16)
    med = np.median(stack, axis=0)
    dev = np.abs(stack - med).max(axis=0)
    return dev <= tol, med


def camera_drift(grays, sample_indices, static):
    """Background-only camera motion, via ORB + RANSAC homography.

    Phase correlation is not usable here: on a large uniform belt a moving box
    dominates the correlation surface and gets reported as a global shift, which
    produces a confident and completely wrong verdict. Restricting to keypoints
    that fall on pixels the median-image background says never change removes both the moving
    objects AND the scrolling belt texture from the estimate.

    `static` is the boolean mask from static_mask(). Keypoints are kept only where BOTH the base
    frame's and the current frame's keypoint land on a static pixel, so a match cannot pair one
    fixed surface with one scrolling patch of belt.

    Returns (median_drift_px, n_samples_used). Fewer than 8 usable samples means
    the footage cannot answer the question, which is reported as uncertainty
    rather than guessed at.
    """
    h, w = static.shape

    def keep_static(kp, des):
        if des is None or len(kp) == 0:
            return [], None
        pts = np.array([k.pt for k in kp], np.float32)
        xi = np.clip(pts[:, 0].astype(int), 0, w - 1)
        yi = np.clip(pts[:, 1].astype(int), 0, h - 1)
        keep = static[yi, xi]
        if keep.sum() == 0:
            return [], None
        return [k for k, ok in zip(kp, keep) if ok], des[keep]

    orb = cv2.ORB_create(nfeatures=1500)
    base_gray = grays[0]
    base_kp, base_des = orb.detectAndCompute(base_gray, None)
    base_kp, base_des = keep_static(base_kp, base_des)
    if base_des is None or len(base_kp) < 30:
        return 0.0, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    drifts = []
    for i in sample_indices[1:]:
        kp, des = orb.detectAndCompute(grays[i], None)
        kp, des = keep_static(kp, des)
        if des is None or len(kp) < 10:
            continue
        knn = matcher.knnMatch(base_des, des, k=2)
        good = []
        for pair in knn:
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 12:
            continue
        src = np.float32([base_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None or inliers is None:
            continue
        inl = src[inliers.ravel() == 1]
        if len(inl) < 8:
            continue
        moved = cv2.perspectiveTransform(inl, H)
        if moved is None:
            continue
        d = np.linalg.norm(moved.reshape(-1, 2) - inl.reshape(-1, 2), axis=1)
        drifts.append(float(np.median(d)))

    if not drifts:
        return 0.0, 0
    return float(np.median(drifts)), len(drifts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sample", type=int, default=120,
                    help="frames to sample for the drift/motion statistics")
    args = ap.parse_args()

    cap, _ = open_capture(args.video)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    name = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
    size_mb = os.path.getsize(args.video) / 1e6

    print(f"file        : {args.video}  ({size_mb:.1f} MB)")
    print(f"resolution  : {w} x {h}")
    print(f"codec       : {name}   {'*** HEVC - re-export as H.264 MP4 ***' if name.lower() in ('hevc','hvc1') else ''}")
    print(f"fps         : {fps:.2f}")
    print(f"frames      : {total}")
    print(f"duration    : {total / fps if fps else 0:.1f} s")

    n = min(total, args.sample)
    if n == 0:
        sys.exit("  ERROR: video reports zero frames")
    idx = np.unique(np.linspace(0, n - 1, n).astype(int))

    grays = []
    i = 0
    want = set(idx.tolist())
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        i += 1
    cap.release()
    if len(grays) < 2:
        sys.exit("  ERROR: could not decode enough frames to analyse")

    stack = np.stack(grays)
    median = np.median(stack, axis=0).astype(np.uint8)

    fg = [np.count_nonzero(np.abs(g.astype(np.int16) - median) > 30) / g.size for g in stack]
    fg_pct = np.array(fg) * 100

    static, _ = static_mask(grays)
    drift_px, n_drift = camera_drift(grays, range(len(grays)), static)

    comps = []
    for g in stack:
        m = (np.abs(g.astype(np.int16) - median) > 30).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        nlab, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        big = sum(1 for i in range(1, nlab) if stats[i, cv2.CC_STAT_AREA] > 0.0005 * g.size)
        comps.append(big)
    comps = np.array(comps)

    train_n = int(total * 0.6)
    dur = total / fps if fps else 0.0
    print("\n--- what the pipeline needs from this footage ---")

    ok = True

    def check(label, good, detail):
        nonlocal ok
        mark = "PASS" if good else "WARN"
        if not good:
            ok = False
        print(f"  [{mark}] {label:34} {detail}")

    check("objects are moving", 0.3 < fg_pct.mean() < 60.0,
          f"{fg_pct.mean():.2f}% of pixels change per frame (0 = nothing moves)")
    check("enough foreground signal", fg_pct.max() > 1.0,
          f"peak {fg_pct.max():.1f}%  (needs >1% to segment at all)")
    check("resolution workable", w >= 1280,
          f"{w}px wide  ({'fine' if w >= 1920 else 'OK, but 1920x1080 gives smaller boxes'})")
    check("clip long enough overall", dur >= 10.0,
          f"{dur:.1f} s  (want >= 10 s so a box can traverse belt and reach the test tail)")
    check("held-out tail long enough", dur * 0.4 >= 4.0,
          f"last 40% = {dur * 0.4:.1f} s / {int(total * 0.4)} frames - a box needs ~3-4 s "
          f"to cross, so a track that starts too late never reaches it and is reported n/a")
    check("frame rate usable", 20 <= fps <= 120,
          f"{fps:.1f} fps  (below 20 optical flow struggles; 50-60 is ideal)")

    if n_drift < 8:
        print("  [WARN] camera static (undecidable)  only "
              f"{n_drift} usable samples - too little static background texture.\n"
              "         Judge by hand: compare a frame from the start with one from the end.\n"
              "         If a belt roller or floor marking has moved, the camera was not static.")
    else:
        check("camera is static", drift_px < 4.0,
              f"median background shift {drift_px:.2f} px over {n_drift} sampled frames, measured on "
              f"{int(static.sum() * 100 / static.size)}% of the frame that never changes "
              f"(a handheld camera is 10-100x this)")

    print(f"\n  foreground blobs per frame: median {np.median(comps):.0f}, "
          f"min {comps.min()}, max {comps.max()}")
    if np.median(comps) < 2:
        print("  [WARN] never more than 1 object on the belt - the assignment needs 2+ boxes\n"
              "         touching at some point, plus 3+ types for Module E.")
    elif np.median(comps) > 8:
        print("  [WARN] very many separate blobs - the belt may be cluttered, or the\n"
              "         background model is picking up shadows and belt texture as objects.")

    print("\n--- these three cannot be checked automatically, confirm by eye ---")
    print("  1. a known-size reference object lying on the belt (A4 sheet = 29.7 x 21.0 cm,\n"
          "     or a bank card = 8.56 x 5.40 cm) - Module B needs it for px->cm")
    print("  2. at least two boxes touching/overlapping somewhere - Module A's watershed\n"
          "     result is specifically about separating them")
    print("  3. at least 3 visibly different box types for Module E")
    print("\n  Also: 'shooting it yourself' means the 15 checkerboard photos must come from\n"
          "  the SAME phone at the SAME height, or the cm numbers will not match.")

    print(f"\nverdict: {'usable as-is' if ok else 're-shoot recommended before running the modules'}")


if __name__ == "__main__":
    main()
