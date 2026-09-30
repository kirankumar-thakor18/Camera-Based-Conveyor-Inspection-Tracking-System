"""
Helper for Module E: cut upright crops of every segmented box out of your video.
Crops land in module_E_recognition/crops/unsorted/ - then MOVE them by hand into one folder per box type,
e.g. crops/small_brown/, crops/big_white/, crops/printed/  (>= 3 types, ~10+ crops each).

Run: python module_E_recognition/extract_crops.py --video data/conveyor.mp4 --every 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from common import estimate_background, read_video
from module_A_segmentation.segmentation import segment_boxes
from module_E_recognition.recognize import upright_crop

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--out", default=str(Path(__file__).parent / "crops" / "unsorted"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    frames, _, _ = read_video(a.video, resize_width=640)
    bg = estimate_background(frames)
    H, W = frames[0].shape[:2]
    n = 0
    for i in range(0, len(frames), a.every):
        for d in segment_boxes(frames[i], bg):
            x, y, w, h = cv2.boundingRect(d["contour"])
            # margin 3 - must match pipeline.touches_border so that a query crop handed to the
            # recogniser is framed exactly like the training crops
            if x <= 3 or y <= 3 or x + w >= W - 3 or y + h >= H - 3:  # skip boxes cut by the frame border
                continue
            cv2.imwrite(str(out / f"f{i:04d}_{n:04d}.png"), upright_crop(frames[i], d["box_pts"]))
            n += 1
    print(f"Saved {n} crops to {out}. Now sort them into one sub-folder per box type.")
