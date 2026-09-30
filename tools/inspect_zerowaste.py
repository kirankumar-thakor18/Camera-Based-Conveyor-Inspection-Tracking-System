"""Inspect an extracted ZeroWaste archive and report exactly what this project can use.

ZeroWaste ships no video files - only extracted frames plus annotation images - so the pipeline
cannot be pointed at it directly.  This script does the discovery that has to happen before the
conversion in `convert_zerowaste.py` can be written against reality instead of guesses.

Run this FIRST, after unzipping, and read the output:

    python tools/inspect_zerowaste.py data/zerowaste/zerowaste-f
    python tools/inspect_zerowaste.py            # auto-detects, use this if unsure

It answers four questions the converter needs:
  1. Where are the RGB frames (as opposed to the masks)?
  2. Which material classes are actually present, and how many objects of each?
  3. What is the frame-number stride inside each sequence?  ZeroWaste annotated a
     tracking subset every 10th frame and a diversity subset every 100th, so stride is what
     decides whether a folder can be replayed as continuous video and at what fps.
  4. Are any sequences dense enough to be worth converting at all?
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("OpenCV is required: pip install -r requirements.txt")

# ZeroWaste semantic-segmentation palette: 0 = paper/background, 1..4 = the four material classes.
# Ordered as in the CVPR 2022 paper: cardboard, soft plastic, rigid plastic, metal.
SEMSEG_NAMES = {1: "cardboard", 2: "soft_plastic", 3: "rigid_plastic", 4: "metal"}

FRAME_RE = re.compile(r"(\d+)_frame_(\d+)")
SRC_FPS = 120.0  # ZeroWaste-f was recorded at 120 fps before subsampling


MASK_WORDS = ("mask", "seg", "label", "annot", "coco_mask")


def classify_dir(p: Path) -> str:
    """Work out whether a folder holds RGB frames or annotation masks.

    The folder NAME is the reliable signal here.  Pixel counting is only a fallback, because
    ZeroWaste stores its instance masks as JPEG too: JPEG ringing spreads the handful of flat
    palette colours over hundreds of values, so a colour-count test reports a mask as a photo.
    """
    low = p.name.lower()
    if any(w in low for w in MASK_WORDS):
        return "annotation mask (by name)"
    imgs = sorted([*p.glob("*.png"), *p.glob("*.PNG")])
    if imgs:
        im = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED)
        if im is not None:
            cols = len(np.unique(im.reshape(-1, im.shape[2]), axis=0)) if im.ndim == 3 \
                else len(np.unique(im))
            if cols <= 12:
                return f"annotation mask ({cols} flat colours, lossless PNG)"
    return "RGB photo (by name)"


def describe_folder(p: Path) -> dict:
    files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in
             {".png", ".jpg", ".jpeg", ".bmp"}]
    seqs = collections.defaultdict(list)
    unparsed = 0
    for f in files:
        m = FRAME_RE.search(f.name)
        if m:
            seqs[m.group(1)].append(int(m.group(2)))
        else:
            unparsed += 1
    return {"dir": p, "n": len(files), "seqs": seqs, "unparsed": unparsed}


def find_zerowaste_root(start: Path) -> Path | None:
    """Locate a directory that has split subfolders (train/val/test) with frames in them."""
    if (start / "train").is_dir() and any((start / "train").iterdir()):
        return start
    for cand in sorted(start.rglob("*")):
        if cand.is_dir() and (cand / "train").is_dir() and \
                any(x.is_dir() for x in (cand / "train").iterdir()):
            return cand
    return None


def report(root: Path) -> int:
    print(f"ZeroWaste root: {root}\n")
    splits = [d for d in sorted(root.iterdir()) if d.is_dir()]
    print(f"{'folder':<58} {'files':>7}  kind")
    print("-" * 96)
    folders = []
    for sp in splits:
        for sub in sorted(sp.iterdir()):
            if sub.is_dir():
                info = describe_folder(sub)
                folders.append((sp.name, info))
                print(f"{str(sub.relative_to(root)):<58} {info['n']:>7}  {classify_dir(sub)}")
        if sp.is_dir() and not any(sp.iterdir()):
            print(f"{str(sp.relative_to(root)):<58} {'0':>7}  (empty)")
    print()

    # ---- stride / replayability
    # Only the RGB folders matter for replaying video; the mask folders hold the same filenames,
    # so analysing them too would print every sequence three times.
    rgb_folders = [(sp, info) for sp, info in folders
                   if not any(w in info["dir"].name.lower() for w in MASK_WORDS)]
    if not rgb_folders:
        rgb_folders = folders
        print("NOTE: no clearly-named RGB folder, falling back to every folder.\n")

    print("sequence analysis (stride = gap between consecutive annotated frames)")
    print(f"{'split/folder':<40} {'seq':>4} {'frames':>7} {'stride':>7} {'implied fps':>12}  verdict")
    print("-" * 96)
    best = []
    seen: set[tuple[str, str]] = set()
    for split, info in rgb_folders:
        for seq, nums in sorted(info["seqs"].items()):
            if (split, seq) in seen:
                continue
            seen.add((split, seq))
            nums = sorted(nums)
            if len(nums) < 2:
                print(f"{info['dir'].relative_to(root)!s:<40.40} {seq:>4} {len(nums):>7} "
                      f"{'-':>7} {'-':>11}  only {len(nums)} frame(s), unusable")
                continue
            gaps = np.diff(nums)
            stride = int(np.median(gaps))
            regular = float(np.mean(gaps == stride))
            fps = SRC_FPS / stride if stride else float("nan")
            # a 10-stride sequence replays as continuous 12 fps video; a 100-stride one is
            # a sparse diversity sample and would jump ~0.8 s between frames.
            if stride <= 12 and regular > 0.9:
                verdict = "CONTINUOUS - usable as video"
            elif stride <= 20:
                verdict = "coarse but usable"
            else:
                verdict = "too sparse for optical flow / tracking"
            if stride <= 12:
                best.append((len(nums), split, seq, stride, fps))
            print(f"{info['dir'].relative_to(root)!s:<40.40} {seq:>4} {len(nums):>7} "
                  f"{stride:>7} {fps:>11.2f}  {verdict} (regularity {regular:.0%})")
    print()

    # ---- class histogram
    print("material classes present (from any sem_seg folder found)")
    found_counts = collections.Counter()
    for split, info in folders:
        if "sem_seg" not in info["dir"].name.lower():
            continue
        for f in sorted(info["dir"].glob("*"))[:400]:
            m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim == 3:
                m = m[..., 0]
            vals, counts = np.unique(m, return_counts=True)
            for v, c in zip(vals, counts):
                found_counts[int(v)] += int(c)
        break  # one folder is enough for the class census
    if found_counts:
        total = sum(found_counts.get(v, 0) for v in SEMSEG_NAMES)
        print(f"  {'label':<16} {'name':<16} {'pixel share':>12}")
        print("  " + "-" * 44)
        for v in (0, 1, 2, 3, 4):
            share = found_counts.get(v, 0) / max(total + found_counts.get(0, 0), 1)
            name = "paper/background" if v == 0 else SEMSEG_NAMES.get(v, "?")
            print(f"  {v:<16} {name:<16} {share:>11.2%}")
    else:
        print("  (no sem_seg folder matched, or masks could not be read)")
    print()

    if best:
        best.sort(reverse=True)
        print("best convertible sequences (longest first):")
        for n, split, seq, stride, fps in best[:8]:
            print(f"  {split}/{seq}_frame_*  {n} frames  stride {stride}  -> "
                  f"{src_fps_str(stride)} at {fps:.2f} fps")
        print("\nNext:  python tools/convert_zerowaste.py --root <root> --seq "
              f"{best[0][1]}/{best[0][2]}_frame_*")
    return 0


def src_fps_str(stride: int) -> str:
    return f"{SRC_FPS:.0f} fps source / {stride}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="data",
                    help="folder containing the extracted ZeroWaste-f (default: data)")
    a = ap.parse_args()
    start = Path(a.path)
    if not start.is_dir():
        sys.exit(f"not a folder: {start}")
    root = find_zerowaste_root(start)
    if root is None:
        print(f"No ZeroWaste-f layout found under {start}.")
        print("Expected something like  <...>/zerowaste-f/{train,val,test}/<image folders>/.")
        print("Pass the folder that directly contains train/ val/ test/ .")
        return 1
    return report(root)


if __name__ == "__main__":
    raise SystemExit(main())