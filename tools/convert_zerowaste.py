"""Convert an extracted ZeroWaste release into this project's expected inputs.

ZeroWaste ships no video: it ships extracted frames plus two annotation images per frame
(`sem_seg` = class index map, `coco_mask` = colour-coded instance masks).  Everything downstream
here wants a video plus labelled crops, so this script bridges the two.

What it produces:
  data/conveyor.mp4                 the chosen sequence replayed at its true timing
  data/zerowaste_gt/                the matching ground-truth class maps, for checking Module A
  data/zerowaste_crops/<class>/     one folder per material class, already sorted

The crops come out pre-sorted because `sem_seg` already carries the class label, which is the one
thing the shoot-it-yourself path leaves to a human.

Run `tools/inspect_zerowaste.py` first: it tells you which sequence has enough frames to be worth
converting, and this script takes that answer as `--seq`.

    python tools/convert_zerowaste.py --root data/zerowaste/zerowaste-f
    python tools/convert_zerowaste.py --root data/zerowaste/zerowaste-f --split train --seq 03
    python tools/convert_zerowaste.py --root ... --seq 03 --width 960 --max-frames 400
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Sem-seg palette of ZeroWaste-f.  0 is paper, which the dataset treats as background, so it is
# deliberately not a class here.
CLASS_NAMES = {1: "cardboard", 2: "soft_plastic", 3: "rigid_plastic", 4: "metal"}
SRC_FPS = 120.0
FRAME_RE = re.compile(r"(\d+)_frame_(\d+)")
MASK_WORDS = ("mask", "seg", "label", "annot")

ROOT = Path(__file__).resolve().parent.parent


def frame_key(p: Path):
    """(sequence, frame_number) for a ZeroWaste filename, or None if it does not match."""
    m = FRAME_RE.search(p.name)
    return (m.group(1), int(m.group(2))) if m else None


def list_images(d: Path) -> list[Path]:
    out = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.PNG", "*.bmp"):
        out.extend(d.glob(ext))
    return sorted(out)


def find_role_folder(split_dir: Path, want_semseg: bool) -> Path | None:
    """Locate the sem_seg folder / the RGB photo folder inside a split directory.

    Written defensively on purpose: the role is taken from the folder name, because the
    annotation masks are stored as JPEG and so cannot be told from photos by colour count alone.
    """
    cands = [d for d in sorted(split_dir.iterdir()) if d.is_dir() and list_images(d)]
    named = [d for d in cands if any(w in d.name.lower() for w in MASK_WORDS)]
    if want_semseg:
        exact = [d for d in named if "sem_seg" in d.name.lower() or d.name.lower() == "seg"]
        pool = exact or named
    else:
        pool = [d for d in cands if d not in named]
    if not pool:
        return None
    # among equally plausible names prefer the one holding the most frames
    return max(pool, key=lambda d: len(list_images(d)))


def connected_boxes(sem: np.ndarray, min_area: int, max_frac: float):
    """Per-class connected components as (class_id, x, y, w, h, area)."""
    out = []
    h, w = sem.shape
    limit = max_frac * h * w
    for cid in sorted(set(np.unique(sem).tolist()) - {0}):
        cid = int(cid)
        if cid not in CLASS_NAMES:
            continue
        n, _, stats, _ = cv2.connectedComponentsWithStats((sem == cid).astype(np.uint8), 8)
        for i in range(1, n):  # 0 is background
            x, y, cw, ch, area = (int(v) for v in stats[i])
            if min_area <= area <= limit:
                out.append((cid, x, y, cw, ch, area))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/zerowaste/zerowaste-f",
                    help="folder that directly contains train/ val/ test/")
    ap.add_argument("--split", default=None, help="train|val|test (default: whichever has the most)")
    ap.add_argument("--seq", default=None, help="sequence id, e.g. 03 (default: longest)")
    ap.add_argument("--out-video", default="data/conveyor.mp4")
    ap.add_argument("--out-crops", default="data/zerowaste_crops")
    ap.add_argument("--out-gt", default="data/zerowaste_gt")
    ap.add_argument("--width", type=int, default=None,
                    help="downscale frames to this width (1920 native is slow to process)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--margin", type=int, default=3, help="crop border, px, matching extract_crops.py")
    ap.add_argument("--min-area", type=int, default=2500, help="drop specks below this area, px")
    ap.add_argument("--max-area-frac", type=float, default=0.25,
                    help="drop anything bigger than this fraction of the frame as a bad blob")
    ap.add_argument("--stride-limit", type=int, default=20,
                    help="reject sequences sparser than this (too gappy for optical flow)")
    ap.add_argument("--clean", action="store_true", help="wipe the crop folder first")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir():
        sys.exit(f"No such folder: {root}\nRun tools/inspect_zerowaste.py to find the right path.")

    # ---- pick the split
    splits = [d for d in (root / s for s in ("train", "val", "test")) if d.is_dir()]
    if not splits:
        sys.exit(f"{root} has no train/ val/ test/ subfolders - is this the right folder?")
    if a.split:
        splits = [d for d in splits if d.name == a.split]
        if not splits:
            sys.exit(f"--split {a.split} not present")

    # ---- pick split+sequence by frame count
    best = None
    for sp in splits:
        rgb = find_role_folder(sp, want_semseg=False)
        if rgb is None:
            continue
        seqs: dict[str, list[int]] = {}
        for f in list_images(rgb):
            k = frame_key(f)
            if k:
                seqs.setdefault(k[0], []).append(k[1])
        for seq, nums in seqs.items():
            if a.seq is not None and seq != a.seq:
                continue
            if len(nums) < 3:
                continue
            nums.sort()
            stride = int(np.median(np.diff(nums)))
            if stride > a.stride_limit:
                continue
            cand = (len(nums), sp.name, seq, rgb, stride)
            if best is None or cand[0] > best[0]:
                best = cand
    if best is None:
        sys.exit("No usable sequence found. Run tools/inspect_zerowaste.py to see why - "
                 "sequences with stride > %d are rejected as too sparse for optical flow."
                 % a.stride_limit)

    n_frames, split_name, seq, rgb_dir, stride = best
    sem_dir = find_role_folder(root / split_name, want_semseg=True)
    if sem_dir is None:
        sys.exit(f"No sem_seg folder found in {root / split_name} - cannot label crops.")
    fps = SRC_FPS / stride
    print(f"chosen : {split_name}/{seq}_frame_*   {n_frames} frames, stride {stride}")
    print(f"frames : {rgb_dir}")
    print(f"labels : {sem_dir}")
    print(f"replay : {SRC_FPS:.0f} fps source / stride {stride} = {fps:.2f} fps")

    # ---- gather frames
    pairs = []
    for f in list_images(rgb_dir):
        k = frame_key(f)
        if not k or k[0] != seq:
            continue
        tag = f"{seq}_frame_{k[1]:06d}"
        sem = None
        for ext in (".PNG", ".png", ".jpg", ".jpeg"):
            cand = sem_dir / (tag + ext)
            if cand.is_file():
                sem = cand
                break
        pairs.append((k[1], f, sem))
    pairs.sort()
    if a.max_frames:
        pairs = pairs[:a.max_frames]
    if len(pairs) < 3:
        sys.exit("Fewer than 3 usable frames - nothing to convert.")

    # ---- write the clip and the crops
    crops_root = ROOT / a.out_crops if not Path(a.out_crops).is_absolute() else Path(a.out_crops)
    gt_root = ROOT / a.out_gt if not Path(a.out_gt).is_absolute() else Path(a.out_gt)
    vid_path = ROOT / a.out_video if not Path(a.out_video).is_absolute() else Path(a.out_video)
    if a.clean and crops_root.exists():
        shutil.rmtree(crops_root)
    crops_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)
    vid_path.parent.mkdir(parents=True, exist_ok=True)
    for cid in CLASS_NAMES.values():
        (crops_root / cid).mkdir(parents=True, exist_ok=True)

    first = cv2.imread(str(pairs[0][1]))
    if first is None:
        sys.exit(f"unreadable image: {pairs[0][1]}")
    H, W = first.shape[:2]
    ow = min(a.width, W) if a.width else W
    oh = int(round(H * ow / W))
    if (ow, oh) != (W, H):
        print(f"scale  : {W}x{H} -> {ow}x{oh}  ({ow / W:.0%})")

    vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))
    kept: dict[str, int] = {}
    total_boxes = 0
    for idx, (num, fpath, sempath) in enumerate(pairs):
        img = cv2.imread(str(fpath))
        if img is None:
            continue
        sem = None
        if sempath is not None:
            sem = cv2.imread(str(sempath), cv2.IMREAD_UNCHANGED)
            if sem is not None and sem.ndim == 3:
                sem = sem[..., 0]
        if sem is not None and sem.shape[:2] != img.shape[:2]:
            sem = cv2.resize(sem, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        small = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA) if (ow, oh) != (W, H) else img
        vw.write(small)

        if sem is not None:
            s_small = cv2.resize(sem, (ow, oh), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(gt_root / f"{seq}_frame_{num:06d}.png"), s_small)
            boxes = connected_boxes(s_small, a.min_area, a.max_area_frac)
            for cid, x, y, w, h, _ in boxes:
                x0, y0 = max(0, x - a.margin), max(0, y - a.margin)
                x1, y1 = min(ow, x + w + a.margin), min(oh, y + h + a.margin)
                if x1 - x0 < 8 or y1 - y0 < 8:
                    continue
                name = CLASS_NAMES[cid]
                dst = crops_root / name / f"{seq}_{num:06d}_{x:05d}_{y:05d}.png"
                cv2.imwrite(str(dst), small[y0:y1, x0:x1])
                kept[name] = kept.get(name, 0) + 1
                total_boxes += 1
        if idx % 50 == 0:
            print(f"  [{idx + 1}/{len(pairs)}] frames")
    vw.release()

    print(f"\nwrote   {vid_path}  ({len(pairs)} frames @ {fps:.2f} fps, {ow}x{oh})")
    print(f"wrote   {gt_root}  (ground-truth class maps for Module A)")
    print(f"wrote   {crops_root}  ({total_boxes} labelled crops, already sorted)")
    for name in sorted(kept):
        print(f"          {name:<15} {kept[name]}")
    missing = [c for c in CLASS_NAMES.values() if c not in kept]
    if missing:
        print(f"  note  : no crops for {missing} - Module E needs 3+ classes")
    print(f"\nnext    python pipeline.py --video {a.out_video} --crops {a.out_crops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())