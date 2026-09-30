"""
Integration pipeline, in the order the assignment specifies:

    frame stream -> A segment boxes -> D Kalman-track centroids -> B pixels -> cm
                 -> C belt speed from optical flow -> E classify each tracked box's type

Prints per tracked box:  ID | size (cm) | speed (cm/s) | recognised type
and saves outputs/pipeline_out.mp4 + outputs/pipeline_results.csv

Run:
  python pipeline.py --video data/conveyor.mp4 --box-height-cm 5
  (no calibration yet?  add  --px-per-cm 11.5  as a rough fallback)
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, estimate_background, load_calibration, read_video, scale_K
from module_A_segmentation.segmentation import segment_boxes
from module_B_calibration.measure_box import Metric
from module_C_optical_flow.optical_flow import box_flow_speed, lk_pyramid
from module_D_tracking.kalman_tracker import MultiBoxTracker
from module_E_recognition.recognize import (AlignmentClassifier, EigenBoxes, load_dataset, upright_crop)


def touches_border(det, shape, m=3):
    x, y, w, h = cv2.boundingRect(det["contour"])
    return x <= m or y <= m or x + w >= shape[1] - m or y + h >= shape[0] - m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib", default=str(ROOT / "module_B_calibration" / "calibration.npz"))
    ap.add_argument("--crops", default=str(ROOT / "module_E_recognition" / "crops"))
    ap.add_argument("--box-height-cm", type=float, default=None, help="height of the box top face above the belt (needed for a cm/s number)")
    ap.add_argument("--px-per-cm", type=float, default=None, help="fallback when no calibration is available")
    ap.add_argument("--recognizer", choices=["eigen", "align"], default="eigen")
    ap.add_argument("--classify-every", type=int, default=5)
    ap.add_argument("--e-test-frac", type=float, default=0.4,
                    help="held-out fraction of the crops; the recogniser trains only on the first "
                         "(1-frac), so the types reported below are not self-matches")
    ap.add_argument("--min-frames", type=int, default=10, help="only report tracks seen this many frames")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-undistort", action="store_true")
    a = ap.parse_args()

    frames, fps, s = read_video(a.video, max_frames=a.max_frames, resize_width=640)
    H, W = frames[0].shape[:2]
    cal = load_calibration(a.calib)
    if cal is not None and a.px_per_cm is None:
        K, dist = scale_K(cal["K"], s), cal["dist"]
        metric = Metric(K, cal["belt_R"], cal["belt_t"], a.box_height_cm or 0.0)
        if not a.no_undistort:
            frames = [cv2.undistort(f, K, dist) for f in frames]  # Module B: remove lens distortion
        print(f"[B] calibration loaded (f={K[0,0]:.0f}px) - sizes/speeds in cm on the plane {a.box_height_cm} cm above the belt")
        if not a.box_height_cm:
            print("[B] WARNING --box-height-cm not given (assuming 0).  The px->cm scale is set by that "
                  "height, so cm/s and cm sizes will be biased; pass --box-height-cm <cm> for real numbers.")
    elif a.px_per_cm:
        metric = Metric(px_per_cm=a.px_per_cm)
        print(f"[B] NO calibration - using fallback {a.px_per_cm} px/cm (less accurate)")
    else:
        sys.exit("No calibration.npz found. Run module_B_calibration/calibrate.py or pass --px-per-cm.")

    # Module E: train the recogniser on the TRAIN portion of the crops only.
    # Training on 100 % of the crops would make every reported type a self-match: the crops were cut from
    # this very video, so a 1-NN eigenbox lookup would find the query's own pixels and "classify" it
    # perfectly.  Holding out the tail (same temporal protocol as recognize.py) keeps the type column honest.
    rec, names = None, []
    if Path(a.crops).exists():
        train, held, names = load_dataset(a.crops, a.e_test_frac)
        if len(names) >= 3 and train:
            rec = (EigenBoxes() if a.recognizer == "eigen" else AlignmentClassifier()).fit(train)
            print(f"[E] {a.recognizer} recognizer trained on {len(train)} crops "
                  f"(held out {len(held)}), classes = {names}")
        elif len(names) < 3:
            print(f"[E] only {len(names)} type folder(s) - the assignment needs at least 3 box types")
    if rec is None:
        print("[E] no usable crops -> every type prints 'n/a'; run module_E_recognition/extract_crops.py "
              "and sort the crops into <type>/ folders first")
    # Frames whose crops the recogniser was NOT trained on.  Voting on a train-region frame is a
    # self-match, so only the tail contributes to the reported type.
    vote_from = int(len(frames) * (1.0 - a.e_test_frac))

    bg = estimate_background(frames)
    tracker = MultiBoxTracker()
    stats = {}
    vw = cv2.VideoWriter(str(OUT / "pipeline_out.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    prev_g, prev_flow = None, None
    for n, frame in enumerate(frames):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = segment_boxes(frame, bg)                       # A
        tracks = tracker.step(dets)                           # D
        flow = lk_pyramid(prev_g, g) if prev_g is not None else None   # C (hand-made LK)
        vis = frame.copy()
        for t in tracks:
            d = t.det
            st = stats.setdefault(t.id, dict(frames=0, sizes=[], flow_v=[], kf_v=[], votes=[], first=n))
            st["frames"] += 1
            inside = not touches_border(d, frame.shape)
            if inside:                                        # B: pixels -> cm
                st["sizes"].append(metric.box_size_cm(d["box_pts"]))
            c = np.array(d["centroid"])
            v = t.kf.x[2:4]
            st["kf_v"].append(metric.displacement_cm(c, c + v) * fps)
            pm = getattr(t, "mask_prev", None)
            if flow is not None and pm is not None:
                r = box_flow_speed(prev_g, g, pm, fps, metric, *flow)
                if r is not None and r["n_inliers"] >= 8:
                    st["flow_v"].append(r["speed_cm_s"])
            t.mask_prev = d["mask"]
            if rec is not None and inside and n >= vote_from and st["frames"] % a.classify_every == 1:   # E
                p = rec.predict(upright_crop(frame, d["box_pts"]))
                # EigenBoxes always returns a valid index; AlignmentClassifier returns -1 when it could
                # not align the crop to any template.  Indexing names with -1 would silently wrap round
                # to the LAST class, so map it to "unknown" instead.
                st["votes"].append(names[p] if p >= 0 else "unknown")
            if st["votes"]:
                _top, _k = Counter(st["votes"]).most_common(1)[0]
                label = f"{_top} {100*_k//len(st['votes'])}%"
            else:
                label = "n/a"
            cv2.polylines(vis, [d["box_pts"].astype(np.int32)], True, (0, 255, 0), 2)
            sz = np.median(st["sizes"], axis=0) if st["sizes"] else (0, 0)
            sp = np.median(st["flow_v"][-5:]) if st["flow_v"] else 0
            cv2.putText(vis, f"#{t.id} {label}", (int(c[0]) - 40, int(c[1]) - 45), 0, 0.55, (0, 255, 255), 2)
            cv2.putText(vis, f"{sz[0]:.1f}x{sz[1]:.1f}cm {sp:.1f}cm/s", (int(c[0]) - 60, int(c[1]) + 55), 0, 0.5, (255, 255, 255), 1)
        for t in tracker.tracks:  # (B1) refresh mask_prev for EVERY track, unconditionally.
            # tracker.step() only returns tracks that have reached min_hits, so a track that was born
            # one or two frames ago is missing from the loop above and its mask_prev would otherwise stay
            # pinned to its birth frame.  The flow at frame n is measured between n-1 and n, so a mask
            # from n-2 shifts the ROI by ~12px and lets belt/background pixels into the affine fit.
            t.mask_prev = t.det["mask"]
        vw.write(vis)
        prev_g = g
    vw.release()

    # ---------------- report
    rows = []
    print("\n" + "=" * 86)
    print(f"{'ID':>3} | {'frames':>6} | {'size L x W (cm)':>17} | {'speed flow (cm/s)':>17} | {'speed Kalman':>12} | type")
    print("-" * 86)
    for tid, st in sorted(stats.items()):
        if st["frames"] < a.min_frames:
            print(f"{tid:>3} | {st['frames']:>6} | {'-':>17} | {'-':>17} | {'-':>12} | "
                  f"(dropped: seen only {st['frames']} frames, need --min-frames {a.min_frames})")
            continue
        if not st["sizes"]:
            print(f"{tid:>3} | {st['frames']:>6} | {'-':>17} | {'-':>17} | {'-':>12} | "
                  f"(no size: the box never sat fully inside the frame)")
            continue
        L, Wd = np.median(st["sizes"], axis=0)
        if st["flow_v"]:
            fs = np.median(st["flow_v"])
            fs_cell, fs_csv = f"{fs:>17.2f}", round(fs, 2)
        else:
            # The affine fit is only accepted with >= 8 RANSAC inliers, so a short or
            # low-texture track never gets a measurement.  Say that, rather than
            # printing a bare "nan" that looks like a bug.
            fs = float("nan")
            fs_cell, fs_csv = f"{'-':>10} (no flow fit)", ""
        ks = np.median(st["kf_v"][3:]) if len(st["kf_v"]) > 3 else float("nan")
        if st["votes"]:
            typ, k = Counter(st["votes"]).most_common(1)[0]
            share = k / len(st["votes"])
            tail = f"{typ} ({100*share:.0f}% of {len(st['votes'])} held-out votes)"
        else:
            # Distinguish "tried and failed" from "never had a fair frame".  Votes are only collected in the
            # held-out tail, so a track that lived entirely in the train region cannot be classified at all.
            typ, share = "n/a", float("nan")
            tail = ("n/a (track never reached the held-out tail, so any type here would be a self-match)"
                    if rec is not None and st["first"] < vote_from else "n/a (no held-out frames classified)")
        print(f"{tid:>3} | {st['frames']:>6} | {L:>7.2f} x {Wd:<7.2f} | {fs_cell} | {ks:>12.2f} | {tail}")
        rows.append([tid, st["frames"], round(L, 2), round(Wd, 2), fs_csv,
                     round(ks, 2) if not np.isnan(ks) else "", typ,
                     "" if np.isnan(share) else round(share, 2), len(st["votes"])])
    print("=" * 86)
    if rows:
        # r[4] is a float for a real measurement and "" when no affine fit was accepted,
        # so it has to be type-checked before np.isnan is allowed anywhere near it.
        allv = [r[4] for r in rows if isinstance(r[4], (int, float)) and not np.isnan(r[4])]
        nofit = sum(1 for r in rows if r[4] == "")
        if allv:
            print(f"Belt speed (median over boxes, optical flow): {np.median(allv):.2f} cm/s"
                  f"   ({len(allv)} measured, {nofit} track(s) had no usable flow fit)")
    with open(OUT / "pipeline_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "frames", "length_cm", "width_cm", "speed_flow_cm_s", "speed_kalman_cm_s",
                    "type", "vote_share", "n_votes"])
        w.writerows(rows)
    print("Saved", OUT / "pipeline_out.mp4", "and", OUT / "pipeline_results.csv")


if __name__ == "__main__":
    main()
