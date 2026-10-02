# Print it, shoot it, calibrate it

The assignment is explicit: *"Either is acceptable as long as it's **real footage, not synthetic
renders**."* So nothing in this repo generates video any more — you shoot it, and this page is the
recipe. Everything here is also the checklist for your 10-minute demo.

`python tools/make_checkerboard.py` has already produced:

| File | What it is |
|---|---|
| `data/checkerboard/checkerboard_9x6.pdf` | print this, at **100 % / "Actual size"** |
| `data/checkerboard/checkerboard_9x6.png` | same board as an image, for a tablet/laptop screen |

Board is 10x7 squares of 25 mm = 250 x 175 mm, giving **9 x 6 inner corners**. That is what
`--pattern 9x6 --square-mm 25` refers to.

---

## 0. Print and verify the scale

1. Print the **PDF** at 100 %. Not "fit to page", not "shrink to fit".
2. Measure the red bar with a real ruler. It must read **100 mm**.
   - If it reads 98 mm, your print scale is 98 % → use `--square-mm 24.5` (or re-print scaled to 100 %).
   - If it reads 103 mm, use `--square-mm 25.75`.
   - Getting this wrong scales **every centimetre measurement in the whole project**, so check it.
3. Mount the board on something flat and rigid (cardboard, foam board). Never leave it on floppy paper —
   `cornerSubPix` needs a sharp, unwarped corner, and a bent board gives a bad reprojection error.

The PNG works too if you have no printer (verified to still detect at 620x487 px), but paper is better.

---

## 1. Checkerboard photos — the assignment wants 10-15

Shoot these with **the same phone you will film the conveyor with** (Module B recovers the intrinsics of
*that* camera; calibrating one phone and measuring through another is not valid).

Put these in `data/checkerboard/`:

### Easiest: let the tool shoot for you

`tools/shoot_checkerboard.py` drives the camera and saves the shots for you. It watches the live
preview and only saves when the board is **found, large enough, sharp, and a viewpoint you have not
already taken** — which is exactly the set of rules people get wrong by hand.

```bash
python tools/shoot_checkerboard.py --camera 0
```

It asks for the first shot as *lay the board flat on the belt* (saved as `belt_00.png`, the view that
defines the belt plane), then two more flat square-on shots at different distances, then it switches
to asking for a new tilt or roll each time. Keys: `SPACE` save anyway, `g` guide, `d` undo, `r`
reset, `q` quit.

It saves the **raw** camera frame, never the annotated preview, so the overlays cannot corrupt the
corners. When it exits it prints the exact `calibrate.py` command to run next.

### If you would rather shoot by hand

| Shot | How to hold the board | Why |
|---|---|---|
| `belt_00.png` … | **lying FLAT on the belt/table**, filling most of the frame | this is `--belt-image`: its pose defines the belt plane that every cm value is measured against |
| `flat_01.png` | flat, camera square-on, board centred | baseline |
| `flat_02.png` | flat, board shifted to a corner | teaches the optimiser that cx/cy are not at the frame centre |
| `tilt_l_01..03` | flat, camera tilted left / centre / right | focal length vs distance |
| `tilt_u_01..03` | board leaning back, top edge toward the camera | foreshortening → this is what actually pins down the distortion terms |
| `rot_01..03` | flat, board rotated in the image plane | separates fx from fy |
| `near_01..02` | board held close, partly filling the frame | strong perspective; this is where k1/k2 become observable |
| `far_01..02` | board at the far end of the belt | weak perspective, wide baseline |

That is 15. Rules that matter:

- **The whole board must be visible in every shot**, with a plain margin around it.
- **Sharp.** A blurry corner dominates the RMS error more than any other mistake.
- **Different angles and different distances** — 15 near-identical head-on shots give you a badly
  conditioned solve and a focal length you should not trust.
- **Fill the frame.** A board covering 20 % of the image wastes resolution.

Then run it:

```bash
python module_B_calibration/calibrate.py --images data/checkerboard --pattern 9x6 --square-mm 25 \
       --belt-image belt_00.png
```

`RMS reprojection error` should be **under ~0.5 px**. If it is above 1 px, the usual causes are a
mis-measured print scale, a blurred corner, a bent board, or `--pattern` given as squares instead of
inner corners.

> On a single plane of views, `k1`, `k2` and `k3` are not separately observable, so expect `k3` to be
> non-zero even on a good lens. Judge the distortion by how much `cv2.undistort` actually bends the
> frame edges, not by whether each individual coefficient looks small.

---

## 2. The conveyor clip — `data/conveyor.mp4`

The assignment allows exactly this: *"a phone camera pointed at objects moving across a table works
fine as a stand-in conveyor belt."*

**Setup**

- Phone **fixed and static** (a stand, a stack of books, a tripod — nothing handheld). The background
  model is the median of the whole clip, so any camera movement ruins it.
- Point **straight down** at the belt/table from ~50-80 cm. Looking across the belt at a shallow angle
  makes box tops very foreshortened and the size estimates bad.
- Even, diffuse lighting. No sunlight patches, no flicker. Avoid the light directly reflecting off
  glossy tape.
- 30 fps, 1080p if your phone offers it, 5-8 seconds.

**What must be in the shot**

| Requirement | Where it comes from | Do this |
|---|---|---|
| boxes moving steadily, constant speed | Modules C, D | slide/push boxes along the belt in a steady line for the whole clip |
| **at least 3 distinct box types**, ~10 samples each | Module E (recognition) | 3+ visibly different boxes: e.g. small plain brown / tall white / printed-and-taped. Different **shape**, not only different colour |
| **2 boxes touching or slightly overlapping** for part of the clip | Module A (watershed verdict) | place two boxes edge-to-edge, and let them overlap by ~1-2 cm mid-clip |
| **one reference box of known size** | Module B (the four camera models) | measure one box with a ruler before filming, write down L x W x height in cm |
| boxes fully inside the frame for part of the clip | Module D, sizes | don't let every box leave the frame instantly |
| the belt plane visible | Module B (`--belt-image`) | the checkerboard shot from §1 |

**Belt speed you can check.** Time one box end-to-end with your phone's stopwatch while you film
(`speed_cm_s = distance_cm / seconds`). Write it down. That is your ground truth for the Module C
number, and quoting it in the demo is what makes the estimate convincing instead of decorative.

**While you are there, also shoot:**

1. `ref_box.jpg` — the reference box alone on the belt, flat, filling the frame. This is the source of
   the `--ref-px / --ref-cm` numbers for `measure_box.py`.
2. `unknown_box.jpg` — a different box alone on the belt, same framing.
3. A few seconds of **only the empty belt**, so `estimate_background` has clean pixels.

### Check the clip before you build anything on it

```bash
python tools/inspect_video.py data/conveyor.mp4
```

This reads only the video. It reports camera shake (estimated from ORB matches on the *background*
only, so a moving box cannot be mistaken for camera motion), how much of the frame changes per frame,
whether the clip is long enough that boxes survive into the held-out tail, and flags HEVC-in-`.MOV`
clips that OpenCV cannot decode.

If it says `re-shoot recommended`, the verdict is worth acting on — a median-background segmenter
cannot recover from camera movement, and every downstream number inherits the error.

---

## 3. Build the Module E crop set

```bash
python module_E_recognition/extract_crops.py --video data/conveyor.mp4 --every 4
```

Then **sort the crops by hand** into `module_E_recognition/crops/<type>/` — at least 3 folders, 10+
crops each. `recognize.py` refuses to run with fewer than 3 types, and a colour-only difference makes
the task degenerate: give the types different aspect ratios or markings too.

The split is **temporal** (first 60 % train / last 40 % test) on purpose — neighbouring video frames are
near-duplicates, so a random split would leak and inflate your accuracy table.

---

## 4. Run it all

```bash
# A - notebook, so the grid shows up in the write-up
jupyter notebook module_A_segmentation/segmentation_comparison.ipynb
python module_A_segmentation/segmentation.py --video data/conveyor.mp4

# B - calibration, then sizes under all four camera models
python module_B_calibration/calibrate.py --images data/checkerboard --pattern 9x6 --square-mm 25 \
       --belt-image belt_00.png
python module_B_calibration/measure_box.py --ref-px <L> <W> --ref-cm <L_cm> <W_cm> \
       --unk-px <L> <W> --ref-height <h> --unk-height <h>

# C - flow + belt speed
python module_C_optical_flow/optical_flow.py --video data/conveyor.mp4 --box-height-cm <h>

# D - Kalman + occlusion dropout
python module_D_tracking/kalman_tracker.py --video data/conveyor.mp4 --dropout-start 40 --dropout-len 5

# E - recognition accuracy
python module_E_recognition/recognize.py

# everything, chained
python pipeline.py --video data/conveyor.mp4 --box-height-cm <h>
```

`--box-height-cm` is the height of the **top face** of the box above the belt. It sets the px→cm scale,
so leaving it out biases the speed by ~9 %; the scripts warn when it is missing.

---

## 5. What to say in the demo (the assignment lists these explicitly)

Show all five of: the **segmentation comparison grid**, the **calibration undistortion before/after**,
the **optical-flow visualisation**, the **Kalman overlay**, and the **recognition accuracy table**.

Then the numbers you actually measured — ground truth vs estimate:

| Quantity | Your measured value | Reference |
|---|---|---|
| Reference box size (cm) | ______ | ruler |
| Belt speed (cm/s) | ______ | stopwatch |
| Box counts / track IDs | ______ | count by hand |

If any of these disagree badly, the usual suspects are in this order: print scale → `--box-height-cm`
→ camera not actually static → `--belt-image` not the flat-on-belt shot.