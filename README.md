# Camera-Based Conveyor Inspection & Tracking (classical CV)

A fixed overhead camera over a conveyor. Segments boxes, turns pixels into centimetres, measures belt
speed, tracks each box through a Kalman filter, and recognises box types. Classical CV only — no
pretrained deep models, no network training.

Real footage only, nothing synthetic. See [`GETTING_THE_DATA.md`](GETTING_THE_DATA.md) for the data.

## Results actually measured on this footage

Clip: `data/conveyor.mp4` — 1920×1080, H.264, 50 fps, 402 frames, 8.0 s, fixed camera.

| | Result | Output |
|---|---|---|
| A segmentation | 8 methods on 5 frames; watershed separates *touching* boxes, colour-homogeneity methods merge them | `A_segmentation_grid.png` |
| B calibration | ⚠️ **not run** — needs real checkerboard photos (see below). Scale taken from a known-size object instead | *(none)* |
| C optical flow | belt speed **17.48 cm/s**; RANSAC 47/56 inliers; dense LK fails on 70.9 % of windows (aperture problem) | `C_a_lucas_kanade_dense.png`, `C_b_sparse_lk.png`, `C_c_affine_ransac.png` |
| D tracking | mean error **3.88 px**; ±2σ band widens 2.4 → 5.8 px across a 5-frame occlusion (frames 40–44) | `D_kalman_plot.png`, `D_kalman_overlay.mp4` |
| E recognition | eigenboxes **67.7 / 51.6 / 50.0 %**, Hu **43.5 / 41.9 / 45.2 %**, alignment **0 %** (clean / rotated / lighting) | `E_eigenboxes.png`, `E_accuracy.txt` |
| pipeline | 51 tracks kept, 22 typed, median box **7.3 cm** (2.5–22.8), median belt **4.80 cm/s** over 46 measured tracks | `pipeline_out.mp4`, `pipeline_results.csv` |

`C` and the pipeline disagree on belt speed (17.48 vs 4.80 cm/s) because they measure differently:
C fits flow on one selected box at one frame, the pipeline takes a median over every tracked box.
Neither is wrong; do not quote one as "the" belt speed.

### The two results that are weaker than they look

**Module B has no output.** `calibrate.py` needs 10–15 photographs of a checkerboard taken *with the
same camera whose video you are analysing*. This clip came from a public dataset, so that camera is
not available. Nothing here is a stand-in for it. Until those photos exist, every centimetre in this
repo rests on one assumption: an object measured at 357×218 px that is really 30×20 cm, giving
**11.4 px/cm** (anisotropy 1.09). `--px-per-cm` exists so scripts can run without calibration — they
print `[B] NO calibration` rather than pretending the number is calibrated.

**Module E clusters shapes, not materials.** `crops/` holds four folders grouped by unsupervised shape
similarity, so `cluster_0..3` are not box types. There are no labels in the footage. Alignment scores
0 % because the objects are small — crops are 14–209 px with a median of **1** SIFT keypoint, and 93 %
have fewer than 10 — not because the alignment code is broken.

## Setup

```bash
pip install -r requirements.txt
python tools/make_checkerboard.py     # 9x6-inner-corner board -> data/checkerboard/
```

Put the clip at `data/conveyor.mp4` and sanity-check it first:

```bash
python tools/inspect_video.py data/conveyor.mp4
```

Read its output before trusting it. It measures frame-to-frame change over the *whole* clip, so a
moving belt reads as motion — a large number there is not evidence that the camera is moving. You must
confirm the camera is static yourself.

## Step 1 — fix the px→cm scale

If you calibrated the camera, skip to step 2 and pass `--box-height-cm` instead of `--px-per-cm`.
Otherwise mark a known-size object:

```bash
python tools/pick_reference.py --mark          # video path is positional, default data/conveyor.mp4
#  -> reference_marked.png: open it, find the known object, note its number

python tools/pick_reference.py --id 4 --real-cm 30,20
#  -> data/reference.json  (357x218 px = 30x20 cm -> 11.4 px/cm)
```

`--pick X1,Y1 X2,Y2 --real-cm W,H` does the same by hand if the object was not auto-detected. Pass a
different clip as the first argument, e.g. `python tools/pick_reference.py other.mp4 --mark`.

## Step 2 — run the modules

Substitute your own numbers. Every command below is the one that produced the table at the top.

```bash
# A - segmentation
python module_A_segmentation/segmentation.py --video data/conveyor.mp4

# C - optical flow and belt speed
python module_C_optical_flow/optical_flow.py --video data/conveyor.mp4 --px-per-cm 11.4

# D - Kalman tracking  (the dropout flags inject the occlusion; omit for a clean run)
python module_D_tracking/kalman_tracker.py --video data/conveyor.mp4 --dropout-start 40 --dropout-len 5

# E - recognition  (extract, group crops into crops/<label>/, then evaluate)
python module_E_recognition/extract_crops.py --video data/conveyor.mp4 --every 12
python module_E_recognition/recognize.py

# everything at once  (~5-8 min on this clip)
python pipeline.py --video data/conveyor.mp4 --px-per-cm 11.4
```

With a real calibration file instead of `--px-per-cm`:

```bash
# B - only reachable with your own checkerboard photos
python module_B_calibration/calibrate.py --images data/checkerboard --pattern 9x6 --square-mm 25 --belt-image belt_00.png
python module_B_calibration/measure_box.py --ref-px L S --ref-cm L S --unk-px L S --ref-height H

python module_C_optical_flow/optical_flow.py --video data/conveyor.mp4 --box-height-cm H
python pipeline.py --video data/conveyor.mp4 --box-height-cm H
```

`--pattern` is **inner** corners (the generated board is 10x7 *squares* = 9x6 *inner* corners),
`--belt-image` is the shot with the board lying flat on the belt (this defines the belt plane), and
`H` is the height of the box's top face above the belt.

`measure_box.py` without `calibration.npz` runs only the two scale-free models (orthographic, affine)
and prints `nan` for weak-perspective and full-perspective, because those need `f`, `K`, `R`, `t`.
It never invents them:

```bash
python module_B_calibration/measure_box.py --ref-px 357 218 --ref-cm 30 20 --unk-px 200 120 --px-per-cm 11.4
```

## What each module shows

**A – Segmentation.** Eight methods (snake, quadtree split & merge written from scratch, standalone
split / merge, watershed, Felzenszwalb with a region graph, mean shift, N-cut) on the same five frames.
Watershed on a distance transform separates *touching* boxes; the colour-homogeneity methods merge
them, because both group pixels by appearance and touching boxes share one.

**B – Calibration.** `cv2.calibrateCamera` gives K and the distortion coefficients from the checkerboard
photos, with a frame undistorted before/after. `P = K[R|t]` is decomposed by hand with an RQ
factorisation and cross-checked against `cv2.decomposeProjectionMatrix`. Box size is then estimated under
four camera models — orthographic, weak-perspective, affine, full-perspective. Only the last uses K *and*
the measured belt-plane pose, so it is the only one that undoes both the box's unknown depth and its
height above the belt. On the reference object above, orthographic gives 17.25 cm where affine gives
16.81 cm, a 2.5 % gap that a single px/cm factor cannot represent.

**C – Optical flow.** Lucas–Kanade written from scratch (2×2 normal equations per window, coarse-to-fine
warping) against sparse feature LK: dense LK dies on textureless faces (the aperture problem) while sparse
LK only sees corners. A 6-parameter affine flow is fitted with RANSAC (inliers green, outliers red) and
converted to cm/s through the scale.

**D – Kalman tracking.** A hand-written constant-velocity filter (state x, y, vx, vy) tracks each box
centroid from Module A, printing predict → measure → update per frame. Through a 5-frame occlusion it
tracks on prediction alone while the plotted ±2σ band visibly widens: the filter keeps predicting, but its
confidence drops.

**E – Recognition.** SIFT + RANSAC-homography alignment (scored on residual alignment error alone),
eigenboxes (PCA via SVD + 1-NN) and Hu moments, compared on held-out crops under clean, rotated and
lighting-changed conditions. Robustness notes are **printed from the measured accuracy table**, so they
cannot contradict it.

## Notes and limits

- The background is the median of the video, so the **camera must be static** and the boxes must move.
- Sizes assume the top face is `--box-height-cm` above the belt. Wrong height → wrong centimetres, and
  omitting the flag biases belt speed by ~9 %, so every script warns instead of quietly returning a wrong
  number.
- Crops are split temporally — first 60 % train, last 40 % test — because neighbouring video frames are
  near-duplicates and a random split would leak. The pipeline collects recognition votes **only in the
  held-out tail**, so no reported type is a self-match; a track that never reaches the tail is reported as
  `n/a` rather than given a confident but meaningless label.
- Tracks shorter than 10 frames are dropped, and a track whose box never sits fully inside the frame
  reports `-` for size rather than a truncated measurement.
- Module A's watershed conclusion is specifically about *touching* boxes. If the notebook's pair-count
  cell prints 0, the footage cannot support it — re-shoot with two boxes overlapping.
