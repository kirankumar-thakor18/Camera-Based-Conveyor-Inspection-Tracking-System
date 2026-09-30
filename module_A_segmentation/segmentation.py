"""
Module A - Segmentation: find the boxes in every frame.

Methods (all 7 implemented; assignment needs >= 4):
  1 Active contour (snake)         4 Region splitting / region merging (two standalone passes)
  2 Split & merge (quadtree, from scratch)   5 Felzenszwalb graph-based (+ region-adjacency-graph inspection)
  3 Watershed with distance-transform markers   6 Mean shift in (colour + position) space
  7 Normalized Cut on a superpixel graph

`segment_boxes(frame, bg)` (background subtraction + watershed split) is the production
segmenter re-used by Modules C, D, E and pipeline.py.

Run:  python module_A_segmentation/segmentation.py --video data/conveyor.mp4
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from skimage import color, filters, graph, segmentation
from sklearn.cluster import MeanShift, estimate_bandwidth

from common import OUT, estimate_background, read_video

WORK_W = 320  # working width for the comparison grid


# --------------------------------------------------------------------------- production segmenter
def foreground_mask(frame, bg, thr=None):
    a = cv2.GaussianBlur(frame, (5, 5), 0).astype(np.int16)
    b = cv2.GaussianBlur(bg, (5, 5), 0).astype(np.int16)
    d = np.abs(a - b).max(axis=2).astype(np.uint8)
    if thr is None:
        t, _ = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = max(t, 20)
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return m


def watershed_split(mask, frame, rel=0.6):
    """Split touching boxes: per-component distance transform, markers at ridge cores, cv2.watershed."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n, cc = cv2.connectedComponents(mask)
    sure_fg = np.zeros_like(mask)
    for k in range(1, n):
        comp = cc == k
        sure_fg[comp & (dist > rel * dist[comp].max())] = 255
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    unknown = cv2.subtract(cv2.dilate(mask, np.ones((3, 3), np.uint8)), sure_fg)
    markers[unknown == 255] = 0
    markers[mask == 0] = 1  # background label
    markers = cv2.watershed(frame.copy(), markers.astype(np.int32))
    markers[markers < 2] = 0
    return markers  # 0 = background, >=2 = box ids


def segment_boxes(frame, bg, min_area_frac=0.002, mask=None):
    """Return list of dicts: mask, centroid (x,y), rect=((cx,cy),(w,h),angle), box_pts, area."""
    if mask is None:
        mask = foreground_mask(frame, bg)
    lab = watershed_split(mask, frame)
    min_area = min_area_frac * frame.shape[0] * frame.shape[1]
    out = []
    for k in np.unique(lab):
        if k == 0:
            continue
        m = (lab == k).astype(np.uint8) * 255
        area = cv2.countNonZero(m)
        if area < min_area:
            continue
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        rect = cv2.minAreaRect(c)
        out.append(dict(mask=m, contour=c, area=area, centroid=(M["m10"] / M["m00"], M["m01"] / M["m00"]),
                        rect=rect, box_pts=cv2.boxPoints(rect)))
    return out


# --------------------------------------------------------------------------- 1. active contour
def m_snake(bgr, bg):
    from skimage.segmentation import active_contour
    gray = filters.gaussian(color.rgb2gray(bgr[..., ::-1]), 2)
    dets = segment_boxes(bgr, bg)
    out = bgr.copy()
    if not dets:
        return out
    d = max(dets, key=lambda z: z["area"])
    x, y, w, h = cv2.boundingRect(d["contour"])
    t = np.linspace(0, 2 * np.pi, 200)
    # The factor is 0.85, so the initial contour starts slightly INSIDE the box's half-extents and the
    # edge energy pulls it OUT onto the true boundary - it does not shrink onto the edge as one might
    # assume from the word "snake".  A larger factor would start outside, but then the weak background
    # side dominates early and the contour tends to stall before reaching the edge.
    init = np.c_[y + h / 2 + 0.85 * h * np.sin(t), x + w / 2 + 0.85 * w * np.cos(t)]  # (row, col)
    # w_line=0 means there is NO balloon/line (shrink-or-grow) force: the only attraction is the image
    # edge term (w_edge=1).  That is deliberate - it makes the method purely edge-driven rather than
    # letting a line term drag the contour off the box.  The downside is that the initialisation has to
    # be close, and the contour has no way to recover if it crosses a weak edge between two boxes.
    snake = active_contour(filters.gaussian(gray, 1), init, alpha=0.015, beta=10, gamma=0.001,
                           w_line=0, w_edge=1, max_num_iter=400, boundary_condition="periodic")
    cv2.polylines(out, [init[:, ::-1].astype(np.int32)], True, (0, 255, 255), 1)  # BGR: cyan initial
    cv2.polylines(out, [snake[:, ::-1].astype(np.int32)], True, (0, 0, 255), 2)    # BGR: red converged
    return out


# --------------------------------------------------------------------------- 2/4. split & merge
def quadtree_split(gray, var_thr=60.0, min_size=4):
    """REGION SPLITTING: split a block into 4 while its intensity variance > var_thr.

    Blocks carry their true (h, w) so odd sizes are tiled exactly: a naive h = s // 2 split of an
    odd-sized block leaves the leftover row/column unlabelled (it silently keeps label 0 and merges
    into whichever region happens to own that id).
    """
    H, W = gray.shape
    labels = np.full((H, W), -1, np.int32)  # -1 == not assigned yet
    n, stack = 0, [(0, 0, H, W)]
    while stack:
        y, x, h, w = stack.pop()
        hh, ww = h // 2, w // 2
        if hh and ww and h > min_size and w > min_size and gray[y:y + h, x:x + w].var() > var_thr:
            stack += [(y, x, hh, ww), (y, x + ww, hh, w - ww),
                      (y + hh, x, h - hh, ww), (y + hh, x + ww, h - hh, w - ww)]
        else:
            labels[y:y + h, x:x + w] = n
            n += 1
    assert (labels >= 0).all(), "quadtree_split left pixels unassigned"
    return labels


def merge_regions(labels, gray, thr=12.0):
    """REGION MERGING: union adjacent regions whose mean intensity differs < thr (running means)."""
    n = labels.max() + 1
    sums = np.bincount(labels.ravel(), gray.ravel().astype(float), n)
    cnt = np.bincount(labels.ravel(), minlength=n).astype(float)
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    pairs = np.concatenate([
        np.c_[labels[:, :-1].ravel(), labels[:, 1:].ravel()],
        np.c_[labels[:-1, :].ravel(), labels[1:, :].ravel()]])
    pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
    means = sums / np.maximum(cnt, 1)
    order = np.argsort(np.abs(means[pairs[:, 0]] - means[pairs[:, 1]]))  # most similar first
    for a, b in pairs[order]:
        ra, rb = find(a), find(b)
        if ra != rb and abs(sums[ra] / cnt[ra] - sums[rb] / cnt[rb]) < thr:
            parent[rb] = ra
            sums[ra] += sums[rb]
            cnt[ra] += cnt[rb]
    root = np.array([find(i) for i in range(n)])
    _, new = np.unique(root, return_inverse=True)
    return new[labels]


def _labels_to(bgr, lab, size):
    lab_big = cv2.resize(lab.astype(np.int32), size, interpolation=cv2.INTER_NEAREST)
    ov = bgr.copy()
    ov[segmentation.find_boundaries(lab_big, mode="inner")] = (0, 255, 0)
    return ov


def _gray256(bgr, blur=True):
    g = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (256, 256), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g, (3, 3), 0) if blur else g


def m_split_merge(bgr, var_thr=60, merge_thr=14):
    g = _gray256(bgr)
    lab = merge_regions(quadtree_split(g, var_thr), g, merge_thr)
    return _labels_to(bgr, lab, (bgr.shape[1], bgr.shape[0]))


def m_split_only(bgr, var_thr=60):
    # Deliberately UNBLURRED, so the "standalone" split pass is not the same instance as the split phase
    # inside split&merge below (which splits the blurred image).  The point of showing these two as
    # separate columns is to expose what each phase does on its own, and that comparison is only
    # meaningful if the shared preprocessing matches - hence _gray256(..., blur=False) here and a comment
    # rather than a silent difference.  Use blur=True to split the same image split&merge splits.
    return _labels_to(bgr, quadtree_split(_gray256(bgr, False), var_thr), (bgr.shape[1], bgr.shape[0]))


def m_merge_only(bgr, block=8, merge_thr=14):
    g = _gray256(bgr)
    yy, xx = np.mgrid[:256, :256]
    lab0 = ((yy // block) * (256 // block) + (xx // block)).astype(np.int32)  # fixed grid, no splitting
    return _labels_to(bgr, merge_regions(lab0, g, merge_thr), (bgr.shape[1], bgr.shape[0]))


# --------------------------------------------------------------------------- 3. watershed
def m_watershed(bgr, bg):
    lab = watershed_split(foreground_mask(bgr, bg), bgr)
    ov = bgr.copy()
    ov[segmentation.find_boundaries(lab, mode="inner")] = (0, 0, 255)
    return ov


# --------------------------------------------------------------------------- 5. graph based
def m_felzenszwalb(bgr, verbose=True):
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    lab = segmentation.felzenszwalb(rgb, scale=150, sigma=0.8, min_size=60)
    rag = graph.rag_mean_color(rgb, lab)  # region adjacency graph (inspect nodes/edges)
    if verbose:
        n_nodes, n_edges = rag.number_of_nodes(), rag.number_of_edges()
        deg = np.mean([d for _, d in rag.degree()]) if n_nodes else 0.0
        # end="" would run this into the next method's line; keep the trailing space explicit.
        print(f"    [RAG: {n_nodes} nodes / {n_edges} edges, mean degree {deg:.2f}] ")
    return _labels_to(bgr, lab, (bgr.shape[1], bgr.shape[0]))


# --------------------------------------------------------------------------- 6. mean shift
def m_meanshift(bgr, w=80, pos_weight=0.6):
    small = cv2.resize(bgr, (w, int(w * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    yy, xx = np.mgrid[:small.shape[0], :small.shape[1]]
    X = np.c_[lab.reshape(-1, 3), pos_weight * xx.ravel(), pos_weight * yy.ravel()]
    bw = estimate_bandwidth(X, quantile=0.12, n_samples=400)
    ms = MeanShift(bandwidth=max(bw, 1.0), bin_seeding=True).fit(X)
    return _labels_to(bgr, ms.labels_.reshape(small.shape[:2]), (bgr.shape[1], bgr.shape[0]))


# --------------------------------------------------------------------------- 7. normalized cut
def m_ncut(bgr, w=160):
    small = cv2.resize(bgr, (w, int(w * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(small[..., ::-1])
    sp = segmentation.slic(rgb, n_segments=150, compactness=15, start_label=1)
    rag = graph.rag_mean_color(rgb, sp, mode="similarity")
    lab = graph.cut_normalized(sp, rag, thresh=0.0005, num_cuts=10)
    return _labels_to(bgr, lab, (bgr.shape[1], bgr.shape[0]))


METHODS = [
    ("1 Active contour", lambda f, bg: m_snake(f, bg)),
    ("2 Split & merge", lambda f, bg: m_split_merge(f)),
    ("3 Watershed", lambda f, bg: m_watershed(f, bg)),
    ("4a Region split only", lambda f, bg: m_split_only(f)),
    ("4b Region merge only", lambda f, bg: m_merge_only(f)),
    ("5 Felzenszwalb", lambda f, bg: m_felzenszwalb(f)),
    ("6 Mean shift", lambda f, bg: m_meanshift(f)),
    ("7 Normalized cut", lambda f, bg: m_ncut(f)),
]


def pick_frames(frames, n=5):
    # Clip the span to the valid range: on a short clip (len < n) the linspace would produce duplicate or
    # negative indices, which would silently render the same frame twice in the grid.
    m = len(frames)
    if m == 0:
        raise ValueError("no frames")
    idx = np.linspace(0.1 * m, 0.9 * m, min(n, m)).astype(int)
    return np.clip(np.unique(idx), 0, m - 1)


def build_grid(frames, bg, out_path, n=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = pick_frames(frames, n)
    s = WORK_W / frames[0].shape[1]
    small = [cv2.resize(frames[i], None, fx=s, fy=s, interpolation=cv2.INTER_AREA) for i in idx]
    bg_s = cv2.resize(bg, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cols = ["Original"] + [m[0] for m in METHODS]
    fig, ax = plt.subplots(n, len(cols), figsize=(2.6 * len(cols), 1.7 * n + 0.6))
    for r, (fi, f) in enumerate(zip(idx, small)):
        ax[r, 0].imshow(f[..., ::-1])
        ax[r, 0].set_ylabel(f"frame {fi}", fontsize=8)
        for c, (name, fn) in enumerate(METHODS, start=1):
            t0 = time.time()
            print(f"  frame {fi} | {name}", end="", flush=True)
            try:
                ax[r, c].imshow(fn(f, bg_s)[..., ::-1])
            except Exception as e:
                # A silent failure here would ship a plausible-looking grid with a blank panel and no
                # visible error in the notebook (stdout is not displayed there), so the traceback goes
                # into the panel itself.
                ax[r, c].text(0.5, 0.5, "FAILED", ha="center", color="red", fontsize=9, weight="bold")
                ax[r, c].text(0.5, 0.38, f"{type(e).__name__}: {e}"[:60], ha="center", color="red", fontsize=5)
                print(f"  FAILED: {type(e).__name__}: {e}", end="")
                traceback.print_exc()
            print(f"  ({time.time() - t0:.1f}s)")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    for c, name in enumerate(cols):
        ax[0, c].set_title(name, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    print("Saved", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=str(OUT / "A_segmentation_grid.png"))
    a = ap.parse_args()
    frames, fps, _ = read_video(a.video, resize_width=640)
    build_grid(frames, estimate_background(frames), a.out)
