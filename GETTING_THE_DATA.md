# Getting the data

The assignment requires *real footage, not synthetic renders*, so nothing in this repo generates
video. Two paths. **Either way you still shoot the checkerboard** — no public conveyor dataset
publishes camera intrinsics, so Module B's `cv2.calibrateCamera` stage always needs your own 15
photos. See `data/checkerboard/PRINT_AND_SHOOT.md` for the printing and shooting recipe.

---

## Path 1 — ZeroWaste (the dataset used for this submission)

[ZeroWaste (CVPR 2022)](https://doi.org/10.5281/zenodo.6269104) is footage of a real paper-sorting
conveyor at a Materials Recovery Facility in Massachusetts. Licensed CC-BY-4.0.

| Property | Why it matters here |
|---|---|
| Real conveyor, **static** camera | the pipeline's median-background model requires a fixed camera |
| 4 material classes | cardboard / soft plastic / rigid plastic / metal → Module E |
| Ground-truth polygon masks | lets Module A be *measured*, not just eyeballed |
| 1920×1080, recorded at 120 fps | annotated frames are every 10th → replays as 12 fps |

### Download

6.98 GB. Some networks (including shared/campus IPs) get a `403 unusual traffic from your network`
from Zenodo's file endpoint, so a browser download is the reliable route:

```bash
curl -L -o zerowaste-f.zip \
  "https://zenodo.org/records/6269104/files/zerowaste-f.zip?download=1"
# md5 1c79def717335a098d99c8bf01c19abf
unzip -q zerowaste-f.zip -d data/zerowaste
```

Verify the archive before using it:

```bash
python -c "import hashlib;print(hashlib.md5(open('zerowaste-f.zip','rb').read()).hexdigest())"
```

### Inspect, then convert

```bash
python tools/inspect_zerowaste.py data/zerowaste
```

This reports the real folder inventory, the per-sequence frame counts, the stride between annotated
frames, the implied replay fps, and the material-class census. It exists so the conversion is driven
by what the archive actually contains rather than by an assumption about its layout.

The most important thing it tells you: ZeroWaste annotated a **tracking** subset every 10 frames and
a **diversity** subset every 100. Only the first can be replayed as video — in the second, adjacent
annotated frames are 0.83 s apart, which would break optical flow and tracking completely.

```bash
python tools/convert_zerowaste.py --root data/zerowaste/zerowaste-f --seq NN
```

It writes `data/conveyor.mp4` (replayed at the correct timing), `data/zerowaste_gt/` (ground-truth
class maps, so Module A can be scored against real annotations) and `data/zerowaste_crops/<class>/`.

Crops come out **pre-sorted by class**, because ZeroWaste's `sem_seg` already carries the label — this
is the one step that is manual on the shoot-it-yourself path. Pass `--width 960` to keep the frames
manageable (native 1920×1080 is slow to process).

A material may simply not appear in a short window. The converter reports per-class counts and warns
if fewer than three classes were found, since Module E needs at least three.

### Known gap: calibration

ZeroWaste does not release camera intrinsics. The cameras are GoPro Hero 7 units mounted ~100 cm
above the belt, and the released frames were additionally rotated and cropped, so even the
manufacturer's intrinsics would not apply.

So Module B is demonstrated on your own checkerboard session (15 photos, same phone), and for the
dataset frames a known-size reference object supplies the metric scale. An MRF waste stream contains
12 oz beverage cans at 6.6 cm diameter × 12.2 cm tall — a dimensionally fixed, publicly documented
object — which makes a defensible reference; cardboard boxes then play the role of the unknown.

---

## Path 2 — shoot it yourself

`data/checkerboard/PRINT_AND_SHOOT.md` is the full recipe: the print-scale check, the 15-shot
checkerboard list, the conveyor shot list, the Module E crop-sorting step, the run commands, and the
demo checklist.

Validate the clip before building anything on it:

```bash
python tools/inspect_video.py data/conveyor.mp4
```

It checks camera shake (from ORB matches restricted to the background, so box motion is not mistaken
for camera motion), foreground signal, clip length against the 60/40 split, frame rate, and whether
OpenCV can decode the container at all. It cannot judge the reference object, touching boxes or type
diversity — those it lists for you to confirm by eye.

This is the only path where Module B's calibration, Module C's cm/s and the known-size reference box
all come from one consistent camera, so every number in the report is mutually consistent. Roughly
15 minutes of work.

---

## What is and is not committed

`data/checkerboard/`, `data/conveyor.mp4`, `module_B_calibration/calibration.npz`,
`module_E_recognition/crops/` and `outputs/` are produced by running the code, so only the checklist,
the printable board and the source are committed — see `.gitignore`.

The development-time synthetic fixture that this project was regression-tested against is
deliberately **not** included, so that nothing in the submission can be mistaken for synthetic
footage.