"""Establish the px->cm scale from an object of known real size.

Why this tool exists
--------------------
`data/conveyor.mp4` is downloaded footage, so `calibration.npz` does not apply to
it: those photos were taken on a different camera, and the footage's own
intrinsics are unknown. Inventing a scale factor would put fabricated
centimetre numbers in the report, so instead you mark an object whose real
dimensions you know.

OpenCV in this environment has no GUI support (cv2.imshow raises
"The function is not implemented"), so this is done in two steps:

  1.  Save a frame with a coordinate grid drawn on it:
          python tools/pick_reference.py data/conveyor.mp4 --grid --frame 200

  2.  Open reference_grid.png in Paint. The status bar shows the cursor
      position in pixels, so you can read off the corners of your reference
      object. Then:

          python tools/pick_reference.py data/conveyor.mp4 --pick X1,Y1,X2,Y2 \
                 --real-cm W,H

It writes data/reference.json, and prints the pipeline command to use.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

REF_JSON = os.path.join("data", "reference.json")
GRID_PNG = "reference_grid.png"
OUT_PNG = "reference_marked.png"

KNOWN_SIZES = [
    ("A4 sheet      ", "21.0 x 29.7 cm portrait, or 29.7 x 21.0 landscape"),
    ("bank card     ", "5.398 x 8.56 cm"),
    ("CD case       ", "12.1 cm diameter"),
    ("12oz can      ", "6.6 cm diameter, 12.2 cm tall"),
    ("battery AA    ", "1.45 x 5.05 cm"),
    ("laptop        ", "common, but check the exact model"),
]


def grid_png(frame, step=100, label_every=500):
    """Frame with a labelled coordinate grid, so pixels can be read off by eye.

    Drawn *inside* the existing frame - no margin is added - so the coordinates
    read off this image are exactly the coordinates of the original frame. An
    offset banner would silently shift every y value the user reports.
    """
    f = frame.copy()
    h, w = f.shape[:2]
    for x in range(0, w, step):
        major = (x % label_every == 0)
        cv2.line(f, (x, 0), (x, h), (0, 200, 255) if major else (60, 180, 200),
                 2 if major else 1)
        if major:
            cv2.putText(f, str(x), (x + 4, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 140, 255), 2)
    for y in range(0, h, step):
        major = (y % label_every == 0)
        cv2.line(f, (0, y), (w, y), (0, 200, 255) if major else (60, 180, 200),
                 2 if major else 1)
        if major:
            cv2.putText(f, str(y), (4, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 140, 255), 2)
    note = "grid 100px, orange 500px - coordinates match the video frame exactly"
    cv2.rectangle(f, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(f, note, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    return f


def parse_pair(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected two comma-separated numbers, e.g. 412,688")
    return float(parts[0]), float(parts[1])


def mark_candidates(video, frame_idx, out_png=OUT_PNG, min_area_frac=0.002):
    """Draw the largest foreground regions with big numbered labels.

    Reading pixel coordinates off a grid by hand is error-prone - a mis-click
    silently produces a plausible-looking but wrong scale. Instead the candidate
    objects are outlined and numbered, so picking one is a single number, and the
    pixel extent comes from the detector instead of from the mouse.
    """
    sys.path.insert(0, ".")
    from common import estimate_background

    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    warm = []
    stride = max(1, n // 60)
    for i in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            warm.append(f)
        if len(warm) >= 60:
            break
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("  could not read frame %d" % frame_idx)
    if len(warm) < 10:
        sys.exit("  could not gather background frames")

    bg = estimate_background(warm)
    sys.path.insert(0, ".")
    from module_A_segmentation.segmentation import foreground_mask

    mask = foreground_mask(frame, bg)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fsize = frame.shape[0] * frame.shape[1]
    boxes = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area_frac * fsize:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 30 or h < 30:
            continue
        boxes.append((a, x, y, w, h))
    boxes.sort(reverse=True)
    boxes = boxes[:12]

    vis = frame.copy()
    for i, (a, x, y, w, h) in enumerate(boxes, 1):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 5)
        tag = "#%d" % i
        org = (x + 6, max(48, y - 16))
        cv2.putText(vis, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 6)
        cv2.putText(vis, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        cv2.putText(vis, "%dx%d px" % (w, h), (x + 6, y + h + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 4)
        cv2.putText(vis, "%dx%d px" % (w, h), (x + 6, y + h + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

    out = np.zeros((46, vis.shape[1], 3), np.uint8)
    out[:] = (0, 0, 0)
    out = np.vstack([out, vis])
    cv2.imwrite(out_png, out)
    return boxes, out_png


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Mark a known-size object to get the px->cm scale.")
    ap.add_argument("video", nargs="?", default="data/conveyor.mp4")
    ap.add_argument("--frame", type=int, default=200)
    ap.add_argument("--grid", action="store_true",
                    help="save %s with a coordinate grid, then quit" % GRID_PNG)
    ap.add_argument("--pick", type=parse_pair, nargs=2, metavar=("X1,Y1", "X2,Y2"),
                    help="top-left and bottom-right pixel coords of the object, as read off the grid")
    ap.add_argument("--real-cm", type=str, metavar="W,H",
                    help="real width and height of that object in cm, e.g. 29.7,21.0")
    ap.add_argument("--out", default=REF_JSON)
    ap.add_argument("--mark", action="store_true",
                    help="outline and number the candidate objects, then quit")
    ap.add_argument("--id", type=int, help="which numbered object is the known-size one")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frame >= n:
        sys.exit("  frame %d is out of range (clip has %d frames)" % (args.frame, n))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("  could not read frame %d" % args.frame)

    h, w = frame.shape[:2]
    print("frame %d of %d,  %dx%d" % (args.frame, n, w, h))

    if args.mark:
        boxes, png = mark_candidates(args.video, args.frame)
        print("\nwrote %s" % png)
        if not boxes:
            print("no candidate objects found on this frame - try another --frame")
            return
        print("\ncandidate objects, largest first:")
        for i, (a, x, y, bw, bh) in enumerate(boxes, 1):
            print("  #%-2d  %4d x %4d px   at (%4d, %4d)   %5.2f%% of frame"
                  % (i, bw, bh, x, y, 100.0 * a / (w * h)))
        print("\npick the one whose real size you know, then run:")
        print("  python tools/pick_reference.py --id N --real-cm W,H --frame %d" % args.frame)
        print("\nknown real sizes, if you are unsure:")
        for a2, b2 in KNOWN_SIZES:
            print("  %s %s" % (a2, b2))
        return

    if args.id is not None:
        boxes, _ = mark_candidates(args.video, args.frame)
        if not (1 <= args.id <= len(boxes)):
            sys.exit("  no object #%d on this frame (found %d)" % (args.id, len(boxes)))
        a, x, y, pw, ph = boxes[args.id - 1]
        if not args.real_cm:
            sys.exit("  object #%d is %d x %d px - now say its real size: "
                     "--real-cm W,H" % (args.id, pw, ph))
        try:
            cm_w, cm_h = parse_pair(args.real_cm)
        except argparse.ArgumentTypeError as e:
            sys.exit("  --real-cm: %s" % e)
        pxcm_x, pxcm_y = pw / cm_w, ph / cm_h
        data = {
            "video": os.path.basename(args.video),
            "frame": args.frame,
            "object_id": args.id,
            "bbox_xywh": [x, y, pw, ph],
            "bbox_wh_px": [pw, ph],
            "real_cm_wh": [cm_w, cm_h],
            "px_per_cm_x": round(pxcm_x, 4),
            "px_per_cm_y": round(pxcm_y, 4),
            "px_per_cm_mean": round((pxcm_x + pxcm_y) / 2, 4),
            "anisotropy": round(pxcm_x / pxcm_y, 3),
        }
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print("\nobject #%d: %d x %d px  =  %.2f x %.2f cm" % (args.id, pw, ph, cm_w, cm_h))
        print("  px/cm across  : %.4f" % pxcm_x)
        print("  px/cm down    : %.4f" % pxcm_y)
        print("  px/cm mean    : %.4f" % data["px_per_cm_mean"])
        print("  anisotropy    : %.3f" % data["anisotropy"])
        if abs(pxcm_x / pxcm_y - 1) > 0.10:
            print("\n  *** axes disagree by %.0f%% - sizes will be approximate."
                  % (abs(pxcm_x / pxcm_y - 1) * 100))
        print("\nwrote %s" % args.out)
        print("\nthen the real pipeline run:")
        print("  python pipeline.py --video %s --px-per-cm %.4f"
              % (args.video, data["px_per_cm_mean"]))
        return

    if args.grid:
        out = grid_png(frame)
        cv2.imwrite(GRID_PNG, out)
        print("\nwrote %s" % GRID_PNG)
        print("\nnext:")
        print("  1. open it:  start  %s" % GRID_PNG)
        print("  2. hover over the TOP-LEFT corner of your known-size object and")
        print("     read the pixel position from the Paint status bar")
        print("  3. same for the BOTTOM-RIGHT corner")
        print("  4. then run, filling in what you read:")
        print()
        print("     python tools/pick_reference.py --pick X1,Y1,X2,Y2 --real-cm W,H")
        print()
        print("known real sizes, if you are unsure:")
        for a, b in KNOWN_SIZES:
            print("  %s %s" % (a, b))
        return

    if not args.pick or not args.real_cm:
        print("\nnothing to do. Either pass --grid to make the marked-up frame,")
        print("or --pick X1,Y1,X2,Y2 --real-cm W,H once you have the coordinates.")
        print("(--frame %d was read to confirm the clip is usable)" % args.frame)
        return

    (x1, y1), (x2, y2) = args.pick
    try:
        cm_w, cm_h = parse_pair(args.real_cm)
    except argparse.ArgumentTypeError as e:
        sys.exit("  --real-cm: %s" % e)
    pw, ph = abs(x2 - x1), abs(y2 - y1)
    if pw < 5 or ph < 5:
        sys.exit("  that box is only %.0fx%.0f px - too small to be the object. re-read the corners."
                 % (pw, ph))
    if cm_w <= 0 or cm_h <= 0:
        sys.exit("  --real-cm must be positive")

    pxcm_x, pxcm_y = pw / cm_w, ph / cm_h
    data = {
        "video": os.path.basename(args.video),
        "frame": args.frame,
        "roi_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "roi_wh_px": [int(pw), int(ph)],
        "real_cm_wh": [cm_w, cm_h],
        "px_per_cm_x": round(pxcm_x, 4),
        "px_per_cm_y": round(pxcm_y, 4),
        "px_per_cm_mean": round((pxcm_x + pxcm_y) / 2, 4),
        "anisotropy": round(pxcm_x / pxcm_y, 3),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

    print("\nwrote %s" % args.out)
    print("  marked size   : %.0f x %.0f px" % (pw, ph))
    print("  real size     : %.2f x %.2f cm" % (cm_w, cm_h))
    print("  px/cm across  : %.4f" % pxcm_x)
    print("  px/cm down    : %.4f" % pxcm_y)
    print("  px/cm mean    : %.4f" % data["px_per_cm_mean"])

    ratio = pxcm_x / pxcm_y
    if abs(ratio - 1) > 0.10:
        print("\n  *** the two axes disagree by %.0f%%." % (abs(ratio - 1) * 100))
        print("      Expected if the object is flat and fronto-parallel. Causes:")
        print("        - perspective: the object is not in the belt plane")
        print("        - the corners you picked are not opposite corners")
        print("        - the object is rotated in frame")
        print("      Sizes will be approximate until this is under ~10%.")

    print("\nthen run the pipeline for real:")
    print("  python pipeline.py --video %s --px-per-cm %.4f"
          % (args.video, data["px_per_cm_mean"]))


if __name__ == "__main__":
    main()
