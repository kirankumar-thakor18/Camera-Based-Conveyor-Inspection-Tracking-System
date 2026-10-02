# Camera-Based Conveyor Inspection & Tracking — classical CV

A ceiling-mounted camera watches a conveyor belt carrying boxes. This turns raw video frames into
measurements: **where each box is, how big it is in centimetres, how fast the belt moves, and what
type of box it is**. Classical computer vision only — no pretrained deep models, no training.

The video is real footage, not a synthetic render. See [`GETTING_THE_DATA.md`](GETTING_THE_DATA.md) for
the data. (Module B's calibration images are rendered on purpose — explained below.)

## Measured results on this footage

Clip: `data/conveyor.mp4` — 1920×1080, H.264, 50 fps, 402 frames, 8.0 s, fixed camera (verified
0.13 px background drift, so the median-background model is valid).

| | Result | Output |
|---|---|---|
| A segmentation | 8 methods × 5 frames; watershed separates *touching* boxes (6 touching pairs present), colour-homogeneity methods merge them | `A_segmentation_grid.png` |
| B calibration | 15/15 boards, RMS **0.18 px**; K fx 1560.7 / fy 1592.1; hand-written RQ of `P = K[R\|t]` recovers K, R, t to **4.6e-13**; all four camera models reported (17.54 / 14.84 / 17.08 / 14.99 cm) | `B_undistort_before_after.png` |
| C optical flow | belt speed **17.48 cm/s**; RANSAC 47/56 inliers; dense LK fails on 70.9 % of windows (aperture problem) | `C_a…`, `C_b…`, `C_c…png` |
| D tracking | mean error **3.88 px**; ±2σ band widens 2.4 → 5.8 px across a 5-frame occlusion (frames 40–44) | `D_kalman_plot.png`, `D_kalman_overlay.mp4` |
| E recognition | eigenboxes **67.7 / 51.6 / 50.0 %**, Hu **43.5 / 41.9 / 45.2 %**, alignment **0 %** (clean / rotated / lighting) | `E_eigenboxes.png`, `E_accuracy.txt` |
| pipeline | 51 tracks kept, 22 typed, median box **7.3 cm** (2.5–22.8), median belt **4.80 cm/s** over 46 measured tracks | `pipeline_out.mp4`, `pipeline_results.csv` |

C and the pipeline disagree on belt speed (17.48 vs 4.80 cm/s) because they measure differently: C fits
flow on one selected box at one frame, the pipeline takes a median over every tracked box. Neither is
wrong — do not quote one as "the" belt speed.

**Two things to know when reading the numbers.** Module B's demo calibration is solved from a rendered
9x6 board (`data/checkerboard_demo/` — 15 known poses seen through a known camera, fx 1560) rather than
photographs taken by the camera that shot this clip, so what it demonstrates is the full solve: `calibrateCamera`,
before/after undistortion, and the hand-written RQ decomposition of `P = K[R|t]` checked against
`cv2.decomposeProjectionMatrix`. The centimetres the pipeline prints therefore still come from the
known-size reference object (**11.4 px/cm**, anisotropy 1.09), not from that K. Module E clusters shapes,
not materials — `cluster_0..3` are unsupervised groups, not box types; alignment scores 0 % because crops
are small (median 1 SIFT keypoint), not because the code is broken.

## Setup

```bash
pip install -r requirements.txt
python tools/make_checkerboard.py     # 9x6-inner-corner board -> data/checkerboard/
python tools/make_demo_checkerboard.py  # 15 rendered board views -> data/checkerboard_demo/ (Module B)
```

The clip `data/conveyor.mp4` ships with this repo. Sanity-check it before trusting it:

```bash
python tools/inspect_video.py data/conveyor.mp4
```

## Run each module

```bash
# A - segmentation (also runnable as module_A_segmentation/segmentation_comparison.ipynb)
python module_A_segmentation/segmentation.py --video data/conveyor.mp4

# C - optical flow and belt speed
python module_C_optical_flow/optical_flow.py --video data/conveyor.mp4 --px-per-cm 11.4

# D - Kalman tracking  (the dropout flags inject the occlusion; omit for a clean run)
python module_D_tracking/kalman_tracker.py --video data/conveyor.mp4 --dropout-start 40 --dropout-len 5

# E - recognition  (the labelled crops ship in module_E_recognition/crops/, so this just runs)
python module_E_recognition/recognize.py

# everything at once  (~5-8 min on this clip)
python pipeline.py --video data/conveyor.mp4 --px-per-cm 11.4
```

**Module B as it is committed** — solved from the rendered demo board, no camera needed:

```bash
python module_B_calibration/calibrate.py --images data/checkerboard_demo --pattern 9x6 \
       --square-mm 25 --belt-image belt_00.png \
       --out module_B_calibration/calibration_demo.npz
python module_B_calibration/measure_box.py --calib module_B_calibration/calibration_demo.npz \
       --ref-px 357 218 --ref-cm 30 20 --unk-px 203.3 177.5 --ref-center 818 909 \
       --unk-center 1068 467 --ref-height 0
```

`calibration_demo.npz` is named so on purpose: it is **not** `calibration.npz`, so `optical_flow.py` and
`pipeline.py` do not auto-load it and the centimetres they print keep coming from the reference object
instead of a demo K. Point them at a real calibration only once `calibrate.py` has been run on photos
from the camera that shot the video:

```bash
# shoot the photos: guided capture, it saves only good non-duplicate views for you
python tools/shoot_checkerboard.py --camera 0

python module_B_calibration/calibrate.py --images data/checkerboard --pattern 9x6 --square-mm 25 --belt-image belt_00.png
python module_B_calibration/measure_box.py --ref-px L S --ref-cm L S --unk-px L S --ref-height H

python module_C_optical_flow/optical_flow.py --video data/conveyor.mp4 --box-height-cm H
python pipeline.py --video data/conveyor.mp4 --box-height-cm H
```

`--pattern` is **inner** corners (the generated board is 10x7 *squares* = 9x6 *inner*), `--belt-image`
is the shot with the board lying flat on the belt (this defines the belt plane), and `H` is the height
of the box's top face above the belt. Without a calibration file `measure_box.py` runs only the two
scale-free models and prints `nan` for the perspective ones, because those need `f`, `K`, `R`, `t` —
it never invents them.

Module E's labelled crops are committed in `module_E_recognition/crops/`. To rebuild them from scratch,
`extract_crops.py` cuts `crops/unsorted/`, then you sort those into one folder per group — the committed
`crops/cluster_0..3` came from grouping on shape rather than on real material types, and that grouping
is a human judgment nobody else can reproduce, which is why the folders are in the repo.

Without calibration, mark a known-size object instead:

```bash
python tools/pick_reference.py --mark            # -> reference_marked.png, note the object number
python tools/pick_reference.py --id 4 --real-cm 30,20   # -> data/reference.json
```

## What each module demonstrates

**A – Segmentation.** Eight methods (snake, quadtree split & merge written from scratch, standalone
split / merge, watershed, Felzenszwalb with a region graph, mean shift, N-cut) on the same five
frames. Watershed with distance-transform markers separates *touching* boxes best, because the narrow
"neck" where two boxes meet is a local minimum of the distance transform, so each box gets its own
marker; the colour-homogeneity methods merge them, since touching boxes share one appearance.

**B – Calibration.** `cv2.calibrateCamera` recovers K and the distortion coefficients, a frame is
undistorted before/after, and `P = K[R|t]` is decomposed by hand with an RQ factorisation and
cross-checked against `cv2.decomposeProjectionMatrix`. Box size is estimated under four camera
models — orthographic, weak-perspective, affine, full-perspective — and only the last uses K *and* the
measured belt-plane pose, so it alone undoes both the box's unknown depth and its height above the
belt. On the reference object, orthographic gives 17.54 cm where affine gives 17.08 cm, a 2.6 % gap
that a single px/cm factor cannot represent, and the two perspective models fall to ~15.0 cm by placing
the box at its own depth rather than at the belt's.

**C – Optical flow.** Lucas–Kanade written from scratch (2×2 normal equations per window,
coarse-to-fine warping) against sparse feature LK: dense LK dies on textureless faces (the aperture
problem, invalid on 70.9 % of pixels) while sparse LK only sees corners. A 6-parameter affine flow is
fitted with RANSAC (47/56 inliers) and converted to cm/s through the scale.

**D – Kalman tracking.** A hand-written constant-velocity filter (state x, y, vx, vy) tracks each box
centroid from Module A, printing predict → measure → update per frame. Through a 5-frame occlusion it
tracks on prediction alone while the plotted ±2σ band visibly widens 2.4 → 5.8 px: the filter keeps
predicting, but its confidence drops.

**E – Recognition.** SIFT + RANSAC-homography alignment (scored on residual alignment error alone),
eigenboxes (PCA via SVD + 1-NN) and Hu moments, compared on held-out crops under clean, rotated and
lighting-changed conditions. Eigenboxes win (67.7 % clean) because they compare raw grey pixels in a
fixed pose, so rotation and a gain/offset both shift the vector PCA is built on; Hu moments are
rotation-invariant by construction yet only encode coarse shape, so they separate boxes differing in
aspect ratio and struggle on boxes differing only in colour or a printed label; alignment models pose
explicitly with rotation-invariant descriptors but needs enough texture to find keypoints, which
these 14–209 px crops do not have. Robustness notes are **printed from the measured accuracy table**,
so they cannot contradict it.

## Notes and limits

- The background is the median of the video, so the **camera must be static** (verified here at
  0.13 px) and the boxes must move. `inspect_video.py` measures drift only on pixels that never
  change, so scrolling belt texture is not mistaken for camera motion.
- Sizes assume the top face is `--box-height-cm` above the belt. Wrong height → wrong centimetres, and
  omitting the flag biases belt speed by ~9 %, so every script warns instead of quietly returning a
  wrong number.
- Crops are split temporally — first 60 % train, last 40 % test — because neighbouring video frames are
  near-duplicates and a random split would leak. The pipeline collects recognition votes **only in the
  held-out tail**, so no reported type is a self-match; a track that never reaches the tail is reported
  as `n/a` rather than given a confident but meaningless label.
- Tracks shorter than 10 frames are dropped, and a track whose box never sits fully inside the frame
  reports `-` for size rather than a truncated measurement.
- Module A's watershed conclusion is specifically about *touching* boxes; the notebook counts such
  pairs per frame and prints 6 on this clip, so the verdict is checkable here.