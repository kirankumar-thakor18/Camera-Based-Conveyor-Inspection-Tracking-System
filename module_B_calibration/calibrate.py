"""
Module B (1/2) - Camera calibration with a checkerboard.

  * cv2.calibrateCamera  -> intrinsic K + radial/tangential distortion
  * undistort a sample frame (before/after image)
  * extrinsics (R, t) of the belt plane from one image, then P = K[R|t] is decomposed BACK into
    K, R, t manually (RQ decomposition) and checked against the ground truth.

Usage:
  python module_B_calibration/calibrate.py --images data/checkerboard --pattern 9x6 --square-mm 25 \
         --belt-image calib_00.png
  (--pattern = INNER corners, e.g. 9x6.  --belt-image = the photo where the checkerboard lies FLAT ON
   THE BELT/TABLE; its pose defines the belt plane used for cm measurements.)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.linalg import rq

from common import OUT, ROOT

np.set_printoptions(precision=4, suppress=True)


# make_checkerboard.py writes the printable board into the same folder the photos go in, and it is a
# perfect frontal render at its own resolution: keeping it would mix two image resolutions into one
# calibrateCamera call and contribute a tilt-free view.  Photos only.
GENERATED = {"checkerboard_9x6.png", "checkerboard_9x6.pdf", "reference_marked.png"}


def find_corners(paths, pattern):
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    found = []                                  # (path, (w, h), refined corners)
    for p in paths:
        if p.name.lower() in GENERATED:
            print(f"  [skip] {p.name} - generated printable board, not a camera photo")
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, c = cv2.findChessboardCorners(g, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            found.append((p, g.shape[::-1], cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit)))
        else:
            print(f"  [skip] no checkerboard found in {p.name}")

    if not found:
        return [], [], None

    # cv2.calibrateCamera needs ONE image size for the whole set, and it must come from an image that
    # actually yielded corners.  Take the resolution most shots agree on, not the first one seen: a
    # single stray file sorting first would otherwise reject every good capture.
    counts = {}
    for _, s, _ in found:
        counts[s] = counts.get(s, 0) + 1
    size = max(counts, key=lambda k: counts[k])
    good, pts = [], []
    for p, s, c in found:
        if s != size:
            print(f"  [skip] {p.name} - {s[0]}x{s[1]} px, but {counts[size]} of {len(found)} detected "
                  f"shots are {size[0]}x{size[1]} px")
            continue
        good.append(p)
        pts.append(c)
    return good, pts, size


def decompose_projection(P):
    """Manual decomposition P = K [R | t]:  P[:, :3] = K R  -> RQ decomposition; fix signs; t = K^-1 p4."""
    M = P[:, :3]
    K, R = rq(M)
    D = np.diag(np.sign(np.diag(K)))  # make focal lengths positive
    K, R = K @ D, D @ R
    scale = K[2, 2]
    K = K / scale
    t = np.linalg.inv(K) @ P[:, 3] / scale
    # R is scaled by K[2,2]; bring it back onto SO(3).  A camera matrix always has det(R) = +1 once
    # diag(K) > 0, so a negative determinant here cannot be fixed by negating R and t together -
    # that would map K[R|t] to -P, not P.  Undo the scale instead and only flip the offending column.
    R = R / max(abs(np.linalg.det(R)) ** (1.0 / 3.0), 1e-12)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return K, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(ROOT / "data" / "checkerboard"))
    ap.add_argument("--pattern", default="9x6", help="inner corners, columns x rows")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--belt-image", default=None, help="file name of the photo with the board flat on the belt")
    ap.add_argument("--out", default=str(Path(__file__).parent / "calibration.npz"))
    a = ap.parse_args()

    cols, rows = map(int, a.pattern.lower().split("x"))
    pattern = (cols, rows)
    sq_cm = a.square_mm / 10.0  # all real-world units in CENTIMETRES
    paths = sorted([p for p in Path(a.images).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.name.lower() not in GENERATED])
    if len(paths) < 10:
        print(f"  [warn] only {len(paths)} image file(s) in {a.images}; the assignment asks for 10-15 "
              f"photos of the board from different angles.")
    good, imgpts, size = find_corners(paths, pattern)
    print(f"Checkerboard found in {len(good)}/{len(paths)} images (need >= 10 for a stable result)")
    if len(good) < 10:
        sys.exit(f"Only {len(good)} usable images - the assignment asks for 10-15 photos taken from "
                 f"different angles. Check --pattern (it is INNER corners: a printed 10x7-square board "
                 f"is --pattern 9x6) and that the whole board is visible and in focus in each shot.")

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq_cm
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([objp] * len(good), imgpts, size, None, None)

    print(f"\nRMS reprojection error: {rms:.4f} px   (good < 0.5 px)")
    print("Intrinsic matrix K =\n", K)
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    print("Distortion [k1 k2 p1 p2 k3] =", dist.ravel())
    per = []
    for i in range(len(good)):
        proj, _ = cv2.projectPoints(objp, rvecs[i], tvecs[i], K, dist)
        per.append(np.linalg.norm(imgpts[i].reshape(-1, 2) - proj.reshape(-1, 2), axis=1).mean())
    print("Per-image mean reprojection error (px):", np.round(per, 3))

    # ---- undistort a sample frame (before / after) -------------------------------------------
    sample = cv2.imread(str(good[0]))
    und = cv2.undistort(sample, K, dist)
    cv2.imwrite(str(OUT / "B_undistort_before_after.png"), np.hstack([sample, und]))
    print("Saved", OUT / "B_undistort_before_after.png", "(left: original | right: undistorted)")

    # ---- extrinsics + manual decomposition of P for the belt-plane image ----------------------
    names = [p.name for p in good]
    # Match on basename so --belt-image can be given either as a bare name or as a path, but never
    # silently fall back to another board: a wrong belt-plane pose changes every cm measurement.
    belt_key = Path(a.belt_image).name if a.belt_image else None
    if belt_key is None:
        bi = 0
        print(f"  [warn] --belt-image not given - using {names[0]} as the belt-plane view.  That photo "
              f"must show the board lying FLAT on the belt/table, otherwise all cm values are wrong.")
    elif belt_key in names:
        bi = names.index(belt_key)
    else:
        sys.exit(f"--belt-image '{a.belt_image}' is not one of the usable boards: {names}.  "
                 f"It must be a shot where the board lies flat on the belt, and the checkerboard must "
                 f"have been detected in it.")
    R, _ = cv2.Rodrigues(rvecs[bi])
    t = tvecs[bi].ravel()
    P = K @ np.c_[R, t]
    print(f"\nExtrinsics for '{names[bi]}':\nR =\n{R}\nt (cm) = {t}")
    print("Projection matrix P = K[R|t] =\n", P)
    K2, R2, t2 = decompose_projection(P)
    print("\nManual RQ decomposition of P:")
    print("  K recovered =\n", K2)
    print("  R recovered =\n", R2)
    print("  t recovered =", t2)
    print(f"  max|K-K'|={np.abs(K - K2).max():.2e}  max|R-R'|={np.abs(R - R2).max():.2e}  max|t-t'|={np.abs(t - t2).max():.2e}")

    # Independent cross-check: OpenCV's own decomposition must agree with the manual RQ above.
    # (Discarding the return value here would leave the claim unverified, so the numbers are compared.)
    out = list(cv2.decomposeProjectionMatrix(P))
    # OpenCV 4 returns (retval, K, R, t, Rx, Ry, Rz); OpenCV 5 dropped the leading retval flag, so the
    # tuple is 7 long in both but shifted by one.  Detect the flag instead of hard-coding an offset.
    if not (isinstance(out[0], np.ndarray) and out[0].shape == (3, 3)):
        out = out[1:]
    Kc = np.asarray(out[0], float).reshape(3, 3)
    Rcv = np.asarray(out[1], float).reshape(3, 3)
    tc = np.asarray(out[2], float).ravel()[:3]
    Kc = Kc / Kc[2, 2]
    dR = Rcv @ R.T                     # RQ is only defined up to a global sign
    if np.trace(dR) < 0:
        dR = -dR
    # Compare K and R only.  OpenCV's transVector lives in the homogeneous P convention and is scaled
    # differently from our metric t, so a component-wise t comparison would be meaningless here - our own
    # max|t-t'| above already checks t, and it is computed in the same units on both sides.
    print(f"  vs cv2.decomposeProjectionMatrix: max|K-K'cv|={np.abs(K - Kc).max():.2e}  "
          f"rotation angle between the two rotations = "
          f"{np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))):.2e} deg")

    # B4: a single plane of views cannot separate k1 from k2 from k3 (they are all radial), so on a
    # real capture you should expect a visibly non-zero k3 even when the lens is close to ideal.
    # Only interpret the numbers above as physical lens distortion if k1 is small and the other terms are not.
    print(f"  note: k1/k2/k3 are not separable from a single plane of checkerboard views - "
          f"expect k3 to be non-zero here; treat the magnitude, not each coefficient, as meaningful.")
    print(f"  camera distance to belt plane along its normal ~ {abs(R[:, 2] @ t):.1f} cm")

    np.savez(a.out, K=K, dist=dist, image_size=np.array(size), rms=rms, belt_R=R, belt_t=t,
             belt_image=names[bi], square_cm=sq_cm)
    print("\nSaved calibration to", a.out)


if __name__ == "__main__":
    main()
