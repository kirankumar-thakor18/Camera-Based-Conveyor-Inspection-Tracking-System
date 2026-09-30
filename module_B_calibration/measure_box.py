"""
Module B (2/2) - real-world box size (cm) under four camera models.

Reference box  : known size (W x L cm, height h_ref) and measured pixel size.
Unknown box    : measured pixel size (and, optionally, its height above the belt).

Models
  orthographic     : ONE magnification s for the whole image, fitted through the origin by least
                     squares on the reference (0 px <-> 0 cm); depth is ignored    size = px / s
  weak perspective : s = f / Z_avg, ONE depth for the whole object (its centroid depth) size = px * Z_avg / f
  affine           : the full linear part A of  p_img = A p_cm + b, least-squares fitted from the 4
                     reference corners, so it absorbs per-axis scale, shear AND image-plane rotation
                     that orthographic cannot.  The unknown box is mapped back through A^-1 and its
                     edges measured in cm.  With --ref-angle 0 and equal px/cm on both axes A is
                     uniform and affine collapses onto orthographic - that is expected, and the
                     program reports it rather than pretending the two models differ.
  full perspective : back-project the 4 top-face corners through K^-1 onto the plane that lies
                     h cm above the belt (plane from the extrinsics) and measure in 3-D.

Usage example (numbers are pixels / cm; angles are degrees of image-plane rotation):
  python module_B_calibration/measure_box.py --ref-px 102 64 --ref-cm 8 5 --unk-px 76 76 \
         --ref-center 320 240 --unk-center 200 300 --ref-height 5 --unk-height 5 --ref-angle 12
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from common import load_calibration

HERE = Path(__file__).parent


# ---------------------------------------------------------------- geometry helpers (also used by pipeline)
def plane_at_height(R, t, h_cm):
    """Plane parallel to the belt, h cm towards the camera.  Returns (n, d) with  n . X = d  (camera coords)."""
    n = R[:, 2].copy()
    d0 = float(n @ t)
    if d0 < 0:
        n, d0 = -n, -d0
    return n, d0 - h_cm


def pixels_to_plane(K, n, d, uv):
    """Intersect the viewing rays of pixels uv (Nx2) with the plane n.X = d. Returns Nx3 camera coords."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    rays = (np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T).T
    lam = d / (rays @ n)
    return rays * lam[:, None]


def size_perspective(K, R, t, corners_px, h_cm):
    """Full-perspective (long, short) side lengths in cm of a box top face given its 4 pixel corners."""
    n, d = plane_at_height(R, t, h_cm)
    P = pixels_to_plane(K, n, d, corners_px)
    s1 = np.linalg.norm(P[0] - P[1])
    s2 = np.linalg.norm(P[1] - P[2])
    return max(s1, s2), min(s1, s2)


def rect_corners(center, wh, angle_deg=0.0):
    """Rectangle corners ordered TL, TR, BR, BL around `center`, optionally rotated in the image."""
    cx, cy = center
    w, h = wh
    c = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return c @ R.T + np.array([cx, cy])


def quad_sides_cm(pts):
    """(long, short) edge lengths of a 4-point quad whose perimeter order is TL, TR, BR, BL."""
    e = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
    return float(e.max()), float(e.min())


def fit_affine_scale(ref_cm_pts, ref_px_pts, ref_px_center):
    """Least-squares fit of A in  p_img = A p_cm  from the 4 reference corner pairs.

    Rows of the arrays are points, so the column form p = A c becomes p_row = c_row @ A.T and the
    least-squares solution for A.T is pinv(cm) @ px_off - hence the transpose on return.  The cm
    corners must have rank 2, i.e. the reference box must not be degenerate.
    """
    if np.linalg.matrix_rank(ref_cm_pts) < 2:
        sys.exit("Degenerate reference box (its cm corners are collinear) - cannot fit an affine map.")
    return (np.linalg.pinv(ref_cm_pts) @ (ref_px_pts - ref_px_center)).T


def affine_size_cm(A, corners_px):
    """Map 4 pixel-space corners back through A^-1 into reference-centimetre space and measure them.

    Takes corners rather than (width, height, angle) on purpose: the image of a rotated rectangle
    under an affine map is a general quadrilateral, not a rectangle, so its axis-aligned bounding
    box plus an angle would not reproduce it.
    """
    if abs(np.linalg.det(A)) < 1e-12:
        sys.exit("Fitted affine matrix is singular - the reference box is degenerate.")
    c = np.asarray(corners_px, float).reshape(-1, 2)
    # forward was px_row = cm_row @ A.T, so the inverse is cm_row = px_row @ inv(A.T) = px_row @ pinv(A).T
    cm = (c - c.mean(0)) @ np.linalg.pinv(A).T
    return np.array(quad_sides_cm(cm))


class Metric:
    """Pixel -> centimetre converter used by Module C and pipeline.py.
    Uses calibration (K, belt plane) if available, else a plain px_per_cm fallback."""

    def __init__(self, K=None, R=None, t=None, box_h_cm=0.0, px_per_cm=None):
        self.K, self.R, self.t, self.h, self.ppc = K, R, t, box_h_cm, px_per_cm
        self.calibrated = K is not None and px_per_cm is None

    def displacement_cm(self, p0, p1):
        if not self.calibrated:
            return float(np.linalg.norm(np.subtract(p1, p0)) / self.ppc)
        n, d = plane_at_height(self.R, self.t, self.h)
        P = pixels_to_plane(self.K, n, d, [p0, p1])
        return float(np.linalg.norm(P[0] - P[1]))

    def box_size_cm(self, corners_px):
        c = np.asarray(corners_px, float)
        if not self.calibrated:
            s1, s2 = np.linalg.norm(c[0] - c[1]), np.linalg.norm(c[1] - c[2])
            return max(s1, s2) / self.ppc, min(s1, s2) / self.ppc
        return size_perspective(self.K, self.R, self.t, c, self.h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default=str(HERE / "calibration.npz"))
    ap.add_argument("--ref-px", type=float, nargs=2, required=True, metavar=("LONG", "SHORT"))
    ap.add_argument("--ref-cm", type=float, nargs=2, required=True, metavar=("LONG", "SHORT"))
    ap.add_argument("--unk-px", type=float, nargs=2, required=True, metavar=("LONG", "SHORT"))
    ap.add_argument("--ref-center", type=float, nargs=2, default=None)
    ap.add_argument("--unk-center", type=float, nargs=2, default=None)
    ap.add_argument("--ref-angle", type=float, default=0.0,
                    help="image-plane rotation of the reference box (deg).  NOTE: this alone will NOT "
                         "separate affine from orthographic, because rotating a rectangle preserves its "
                         "edge lengths.  To see them differ, give a reference whose px/cm differs per axis.")
    ap.add_argument("--unk-angle", type=float, default=0.0, help="image-plane rotation of the unknown box (deg)")
    ap.add_argument("--ref-height", type=float, default=0.0, help="height of reference top face above belt (cm)")
    ap.add_argument("--unk-height", type=float, default=None, help="height of the unknown top face (default = ref)")
    a = ap.parse_args()

    c = load_calibration(a.calib)
    if c is None:
        sys.exit("Run calibrate.py first (calibration.npz not found).")
    K, R, t = c["K"], c["belt_R"], c["belt_t"]
    f = 0.5 * (K[0, 0] + K[1, 1])
    ctr = (K[0, 2], K[1, 2])
    rc = a.ref_center or ctr
    uc = a.unk_center or ctr
    hu = a.ref_height if a.unk_height is None else a.unk_height
    ref_px, ref_cm, unk_px = map(np.array, (a.ref_px, a.ref_cm, a.unk_px))

    res = {}
    # 1 orthographic: ONE scalar magnification for the whole image, least squares through the origin
    #    (minimise sum_i (ref_px_i - s*ref_cm_i)^2 over both reference axes).
    s = float(ref_px @ ref_cm / (ref_cm @ ref_cm))
    res["orthographic"] = unk_px / s
    # 2 weak perspective: s = f / Z_avg, Z_avg = depth of the unknown box centre (top face)
    n, d = plane_at_height(R, t, hu)
    Zavg = pixels_to_plane(K, n, d, [uc])[0, 2]
    res["weak perspective"] = unk_px * Zavg / f
    # 3 affine: full 2x2 linear part fitted on the 4 reference corners.  The reference box is
    #    axis-aligned in ITS OWN cm frame but may appear rotated in the image (--ref-angle); that
    #    rotation and any shear/anisotropy land in A, and the unknown box is mapped back through
    #    A^-1 and measured in cm.
    A = fit_affine_scale(rect_corners((0.0, 0.0), ref_cm), rect_corners(rc, ref_px, a.ref_angle), rc)
    res["affine"] = affine_size_cm(A, rect_corners(uc, unk_px, a.unk_angle))
    # 4 full perspective (needs K, R, t and the top-face height)
    L, S = size_perspective(K, R, t, rect_corners(uc, unk_px, a.unk_angle), hu)
    res["full perspective"] = np.array([L, S])

    print(f"f = {f:.1f} px | reference: {ref_px} px = {ref_cm} cm | unknown: {unk_px} px | Z_avg = {Zavg:.1f} cm")
    print(f"{'camera model':<18}{'long (cm)':>12}{'short (cm)':>12}")
    for k, v in res.items():
        print(f"{k:<18}{v[0]:>12.2f}{v[1]:>12.2f}")
    print(f"\naffine linear part A (px per cm) =\n{np.round(A, 4)}")
    # Rotation / anisotropy / shear must be read off the SINGULAR VALUES of A.  Using the diagonal
    # entries instead reports a 45 deg rotation as "shear 1.41" (2*sin45) and makes anisotropy blow up
    # whenever a diagonal entry crosses zero.  With svals s1 >= s2:
    #   rotation = angle of the leading left singular vector
    #   anisotropy (how much the two px/cm axes disagree) = s1/s2
    #   shear (axis non-orthogonality, rotation removed)  = (s1-s2)/(s1+s2)
    U, svals, _ = np.linalg.svd(A)
    s1, s2 = svals
    scale = float(np.sqrt(s1 * s2))
    aniso = float(s1 / s2)
    rot_deg = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))
    shear = float((s1 - s2) / (s1 + s2))
    print(f"  singular values {np.round(svals, 4)} px/cm | geometric-mean scale {scale:.3f} px/cm | "
          f"rotation absorbed {rot_deg:+.2f} deg | anisotropy {aniso:.4f}x | shear {shear:.2e}")
    if abs(rot_deg) < 0.5 and abs(aniso - 1) < 0.01 and shear < 1e-3:
        print("  -> A is a uniform scale to within 1%, so affine collapses onto orthographic here.")
        print("     That is CORRECT, not a bug: with the reference unrotated, axis-aligned and isotropic")
        print("     there is no rotation or anisotropy left for a full affine map to absorb.  Note that")
        print("     --ref-angle alone will NOT separate them either: rotating a rectangle preserves its")
        print("     edge lengths, so orthographic returns the same long/short sides at every angle.  To see")
        print("     the models genuinely separate you need a reference whose px/cm differs per axis, e.g.")
        print("     --ref-px 200 20 --ref-cm 8 5 (a 10:2.5 px/cm ratio, not 10:1).")
    else:
        print("  -> affine genuinely differs from orthographic: it has absorbed in-image rotation and/or")
        print("     per-axis anisotropy that a single scale factor cannot represent.")

    # Field of view: the assignment asks for the verdict "given the camera's ACTUAL field of view and
    # object distance", so state it in degrees instead of leaving the reader to work it out.
    w_px, h_px = (c["image_size"] if "image_size" in c else (2 * ctr[0], 2 * ctr[1]))
    fov_x = 2 * np.degrees(np.arctan((w_px / 2) / f))
    fov_y = 2 * np.degrees(np.arctan((h_px / 2) / f))
    dist = abs(R[:, 2] @ t)
    covered_x = 2 * dist * np.tan(np.radians(fov_x / 2))
    covered_y = 2 * dist * np.tan(np.radians(fov_y / 2))
    print(f"  field of view {fov_x:.1f} deg x {fov_y:.1f} deg  ->  the belt plane spans roughly "
          f"{covered_x:.0f} cm x {covered_y:.0f} cm at the {dist:.0f} cm camera distance")

    # DISCUSSION (2-3 sentences, as the assignment asks - read the numbers above, then make this your own):
    # Full perspective is the most accurate of the four because it is the only one that uses the
    # calibrated K together with the measured pose (R, t) of the belt plane, so it can undo BOTH the
    # depth of the unknown box and the fact that its top face sits `hu` cm above that plane; with a
    # ~50 deg field of view and the box only a few cm tall at ~60 cm distance the depth relief is small,
    # which is why weak perspective is a close second.  The honest catch is that orthographic and affine
    # ignore depth altogether - orthographic additionally forces BOTH px/cm axes to share ONE scalar,
    # so it is badly biased whenever the reference box is not square in the image (aniso above != 1),
    # which is a modelling error and not a depth error; affine fixes that per-axis scaling but still
    # cannot express the foreshortening that height and depth introduce, which is the whole gap to
    # full perspective.
    print(f"depth relief ratio (box height / camera distance) ~ {hu / dist:.3f}")


if __name__ == "__main__":
    main()
