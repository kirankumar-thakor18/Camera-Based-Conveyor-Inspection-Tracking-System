# Camera-Based Conveyor Inspection & Tracking (classical CV)

A fixed overhead camera over a conveyor. Segments boxes, turns pixels into centimetres, measures belt
speed, tracks each box through a Kalman filter, and recognises box types. Classical CV only — no
pretrained deep models, no training of a network.

The footage is real industrial video (see [`GETTING_THE_DATA.md`](GETTING_THE_DATA.md)); nothing here
is synthetic.

## Setup

```bash
pip install -r requirements.txt
python tools/make_checkerboard.py     # prints a 9x6-inner-corner board to data/checkerboard/
```

Place the real clip at `data/conveyor.mp4` and 15 checkerboard photos in `data/checkerboard/`.

## How to run

| | Command | Output in `outputs/` |
|---|---|---|
| A | `python module_A_segmentation/segmentation.py --video data/conveyor.mp4` (or run `segmentation_comparison.ipynb`) | `A_segmentation_grid.png` |
| B | `python module_B_calibration/calibrate.py --images data/checkerboard --pattern 9x6 --square-mm 25 --belt-image belt_00.png`<br>then `python module_B_calibration/measure_box.py --ref-px L S --ref-cm L S --unk-px L S --ref-height H` | `calibration.npz`, `B_undistort_before_after.png`, 4-model size table |
| C | `python module_C_optical_flow/optical_flow.py --video data/conveyor.mp4 --box-height-cm H` | `C_a/b/c` flow images + cm/s |
| D | `python module_D_tracking/kalman_tracker.py --video data/conveyor.mp4` | `D_kalman_plot.png`, `D_kalman_overlay.mp4` |
| E | `python module_E_recognition/extract_crops.py --video data/conveyor.mp4`, sort crops into `crops/<type>/`, then `python module_E_recognition/recognize.py` | accuracy table, `E_eigenboxes.png` |
| All | `python pipeline.py --video data/conveyor.mp4 --box-height-cm H` | per-box table, `pipeline_out.mp4`, `pipeline_results.csv` |

`--pattern` is **inner** corners (the generated board is 10x7 *squares* = 9x6 *inner* corners);
`--belt-image` is the shot where the board lies flat on the belt, which defines the belt plane;
`H` is `--box-height-cm`, the height of the box's top face above the belt.

## What each module shows

**A – Segmentation.** Eight methods (snake, quadtree split & merge written from scratch, standalone
split / merge, watershed, Felzenszwalb with a region graph, mean shift, N-cut) run on the same five
frames. Watershed on a distance transform separates *touching* boxes; the colour-homogeneity methods
merge them, because both group pixels by appearance and touching boxes share one.

**B – Calibration.** `cv2.calibrateCamera` gives K and the distortion coefficients from the
checkerboard photos, with a frame undistorted before/after. `P = K[R|t]` is then decomposed by hand
with an RQ factorisation and cross-checked against `cv2.decomposeProjectionMatrix`. Box size is then
estimated under four camera models — orthographic, weak-perspective, affine, full-perspective. Only
the last uses K *and* the measured belt-plane pose, so it is the only one that undoes both the box's
unknown depth and its height above the belt, and it is the most accurate.

**C – Optical flow.** Lucas–Kanade written from scratch (2×2 normal equations per window,
coarse-to-fine warping), compared against sparse feature LK: dense LK dies on textureless faces (the
aperture problem) while sparse LK only sees corners. A 6-parameter affine flow is fitted with RANSAC
(inliers green, outliers red) and converted to cm/s through the calibration.

**D – Kalman tracking.** A hand-written constant-velocity filter (state x, y, vx, vy) tracks each box
centroid from Module A, printing predict → measure → update per frame. Through a 5-frame occlusion it
tracks on prediction alone while the plotted ±2σ band visibly widens: the filter keeps predicting,
but its confidence drops.

**E – Recognition.** SIFT + RANSAC-homography alignment (scored on residual alignment error alone),
eigenboxes (PCA via SVD + 1-NN) and Hu moments, compared on held-out crops under clean, rotated and
lighting-changed conditions. The robustness notes are **printed from the measured accuracy table**, so
they cannot contradict it.

## Notes and limits

- The background is the median of the video, so the **camera must be static** and the boxes must move.
- Sizes assume the top face is `--box-height-cm` above the belt. Wrong height → wrong centimetres, and
  omitting the flag biases belt speed by ~9 %, so every script warns instead of quietly returning a
  wrong number.
- Crops are split temporally — first 60 % train, last 40 % test — because neighbouring video frames are
  near-duplicates and a random split would leak. The pipeline collects recognition votes **only in the
  held-out tail**, so no reported type is a self-match against a training crop; a track that never
  reaches the tail is reported as `n/a` rather than given a confident but meaningless label.
- Module A's watershed conclusion is specifically about *touching* boxes. If the notebook's
  pair-count cell prints 0, the footage cannot support it — re-shoot with two boxes overlapping.