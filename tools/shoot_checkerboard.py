"""
Guided checkerboard capture for Module B - so you do not have to know how to take calibration photos.

    python tools/shoot_checkerboard.py

It shows a live camera preview with the checkerboard drawn on top. Move your real printed board (or a
laptop/tablet showing checkerboard_9x6.png) behind the phone until the board fills the dashed guide.
The tool then:

  * locks on only when the board is detected, big enough, sharp and a genuinely NEW viewpoint,
  * counts down 3-2-1 and saves the shot automatically,
  * refuses near-duplicate shots, because 14 copies of one pose calibrate worse than 8 varied ones,
  * tells you what to do next ("tilt left", "move closer", "move up") instead of silently saving junk.

Keys:
    SPACE  force-save the current frame anyway (bypasses the quality gate)
    g      toggle the guide overlay
    d      delete the last saved shot
    r      reset, delete every shot taken in this session
    q/ESC  quit and print the calibrate.py command to run next

Flat-vs-tilted matters, so the first three shots are asked for FLAT (board square to the camera) and the
rest are asked for TILTED. A flat board alone cannot separate k1 from k2 from k3 - all three are radial.

Nothing is calibrated here. This only collects the photos; run module_B_calibration/calibrate.py after.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "checkerboard"

KEY_SPACEBAR, KEY_Q, KEY_ESC = 32, ord("q"), 27

BAR = "=" * 68


def board_guide(w, h, shrink=0.86):
    """Aspect-correct guide box in pixel coords for a 9x6-inner-corner board (10x7 squares)."""
    bw = int(w * shrink)
    bh = int(bw * 7 / 10)
    if bh > int(h * shrink):
        bh = int(h * shrink)
        bw = int(bh * 10 / 7)
    x0 = (w - bw) // 2
    y0 = (h - bh) // 2
    return x0, y0, bw, bh


def draw_text(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def detect(g, pattern):
    ok, c = cv2.findChessboardCorners(g, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(g, c, (11, 11), (-1, -1), crit)


def sharpness(g, corners):
    """Variance of Laplacian inside the board, in px^2.  A soft photo gives a flat surface.

    The ROI is derived from the corner coordinates directly.  cv2.boundingRect is not used here: it
    silently returns a degenerate rect for the float32 point array findChessboardCorners produces,
    which makes the ROI empty and the score read 0 for every frame.
    """
    c = corners.reshape(-1, 2)
    h, w = g.shape
    x0 = int(max(0, c[:, 0].min())) + 6
    y0 = int(max(0, c[:, 1].min())) + 6
    x1 = int(min(w, c[:, 0].max())) - 6
    y1 = int(min(h, c[:, 1].max())) - 6
    if x1 - x0 < 10 or y1 - y0 < 10:
        return 0.0
    roi = g[y0:y1, x0:x1]
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def pose(corners, w, h, pattern):
    """(area_fraction, tilt_deg, roll_deg, cx, cy) for a detected board.

    findChessboardCorners returns corners in row-major order, so the board's four boundary edges can
    be measured directly.

    tilt is perspective foreshortening: viewing the board off-axis shortens the far edge and
    lengthens the near one. roll is in-plane rotation, measured from the top edge's angle to the
    image x-axis. They are reported separately because roll creates NO foreshortening at all (it reads
    0 deg of tilt) yet is just as useful to the optimiser - a rolled board is a valid, distinct view.

    Measured scale, so the thresholds in main() are not guesses: with a board filling ~13 % of a
    1920x1080 frame at ~50 cm, 30 deg of pitch reads ~6 deg of tilt and 30 deg of yaw ~3 deg. Tilt is
    therefore a much weaker signal than roll, which is why both are required to accept a new pose.
    """
    c = corners.reshape(-1, 2)
    grid = c.reshape(pattern[1], pattern[0], 2)  # (rows, cols, xy)

    # Compare opposite edges ONLY. The board is rectangular (9x6 inner corners = 10x7 squares), so the
    # top edge is inherently 9 squares long and the left edge 6 - comparing them to each other would
    # report a constant 33 deg of "tilt" on a perfectly square-on shot. Comparing top-vs-bottom and
    # left-vs-right cancels that aspect ratio and leaves only real perspective foreshortening.
    top = np.linalg.norm(grid[0, -1] - grid[0, 0])
    bottom = np.linalg.norm(grid[-1, -1] - grid[-1, 0])
    left = np.linalg.norm(grid[-1, 0] - grid[0, 0])
    right = np.linalg.norm(grid[-1, -1] - grid[0, -1])

    def asym(a, b):
        return abs(np.degrees(np.arctan2(a - b, a + b)))

    tilt = max(asym(top, bottom), asym(left, right))

    edge = grid[0, -1] - grid[0, 0]
    roll = abs(np.degrees(np.arctan2(edge[1], edge[0])))
    if roll > 90:
        roll = 180 - roll

    xs, ys = c[:, 0], c[:, 1]
    bw = float(xs.max() - xs.min())
    bh = float(ys.max() - ys.min())
    area_frac = float(bw * bh) / float(w * h)
    return area_frac, float(tilt), float(roll), float(xs.mean()), float(ys.mean())


def too_similar(new_pose, saved, min_roll_delta=6.0, min_tilt_delta=1.5, min_move_px=90.0,
                min_scale_delta=0.20):
    """True if this viewpoint duplicates one already saved.

    Calibrating needs the photos to span the intrinsics: ten near-identical shots collapse the
    optimisation onto a single pose and give a worse K than eight genuinely varied ones.

    A shot counts as new if it differs in ANY of rotation, position or scale. Scale is checked because
    the opening shots are deliberately flat and square-on, so rotation is ~0 for all of them and only
    distance and placement can tell them apart - without this, "same board, one step closer" is
    rejected as a duplicate and the flat-shot phase can never be completed.

    Both rotation signals are needed: tilt is numerically weak (see pose()), so relying on it alone
    would call clearly different viewpoints duplicates, while relying on roll alone would accept a
    board that only slid sideways.
    """
    na, nt, nr, ncx, ncy = new_pose
    for a, t, r, cx, cy in saved:
        same_rotation = abs(r - nr) < min_roll_delta and abs(t - nt) < min_tilt_delta
        same_place = abs(cx - ncx) < min_move_px and abs(cy - ncy) < min_move_px
        same_scale = abs(a - na) < min_scale_delta * max(a, na)
        if same_rotation and same_place and same_scale:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--pattern", default="9x6")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--target", type=int, default=15)
    ap.add_argument("--min-area", type=float, default=0.10, help="min board area as a fraction of frame")
    ap.add_argument("--min-sharp", type=float, default=60.0)
    # Thresholds are in the units pose() reports and were calibrated against measured behaviour, not
    # guessed: a square-on board reads 0 tilt / 0 roll, while 30 deg of pitch or yaw reads only ~3-6
    # deg of tilt (see pose's docstring). Roll is the far stronger signal, hence the looser tilt gate.
    ap.add_argument("--flat-tilt", type=float, default=2.0, help="max tilt for the first 3 flat shots")
    ap.add_argument("--flat-roll", type=float, default=4.0, help="max roll for the first 3 flat shots")
    ap.add_argument("--min-tilt", type=float, default=1.5, help="min tilt once flat shots are done")
    ap.add_argument("--min-roll", type=float, default=6.0, help="min roll once flat shots are done")
    ap.add_argument("--cooldown", type=float, default=2.0, help="seconds between auto-captures")
    ap.add_argument("--width", type=int, default=1280, help="requested capture width")
    ap.add_argument("--height", type=int, default=720, help="requested capture height")
    ap.add_argument("--no-preview", action="store_true",
                    help="headless: no cv2.imshow window, status printed to the console instead")
    ap.add_argument("--warmup", type=float, default=0.0,
                    help="seconds to wait before accepting shots, so the board can be positioned first")
    ap.add_argument("--max-seconds", type=float, default=300.0,
                    help="hard stop, so headless mode always terminates on its own")
    args = ap.parse_args()

    cols, rows = map(int, args.pattern.lower().split("x"))
    pattern = (cols, rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(BAR)
    print("  Guided checkerboard capture - Module B")
    print(BAR)
    print(f"  pattern {cols}x{rows} inner corners  ->  board is {(cols + 1)}x{(rows + 1)} squares")
    print(f"  saving to {out}")
    print()
    print("  BEFORE YOU START")
    print("    1. Print data/checkerboard/checkerboard_9x6.pdf at 100 % / 'Actual size'.")
    print("       Measure the red bar with a ruler: it must be 100 mm. If it is not, the whole")
    print("       calibration is wrong by that ratio, so fix the print scale first.")
    print("    2. Stick the board to something flat and RIGID (cardboard). Bent board = bad corners.")
    print("    3. You do NOT need a printer: open checkerboard_9x6.png on a laptop/tablet screen and")
    print("       lay it flat on the belt. It is less sharp than paper but it does work.")
    print("    4. Use the SAME camera for this and for the conveyor video, at the same height.")
    print()
    print("  FIRST SHOT IS SPECIAL - it defines the belt plane, so it is saved as belt_00.png.")
    print("    Lay the board FLAT ON THE MOVING BELT (or the belt surface), camera looking down at it")
    print("    the same way as the conveyor video. Do NOT hold it in your hand at an angle: the rest of")
    print("    the board photos are then free to be tilted, but this one must show the belt itself.")
    print()
    print("  Then hold the board up to the camera and move it as instructed on screen.")
    print("  SPACE=save anyway  g=guide  d=undo  r=reset  q=quit")
    print(BAR)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open camera {args.camera}. Close any app using it (Teams/Zoom/Meet) and retry.")
    # Ask for HD explicitly.  Many webcams hand OpenCV a 640x480 default while happily supplying
    # 720p, and 9x6 inner corners on a board that has to fill the frame need the pixels.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"  camera {args.camera}: {w}x{h}" + (f"  (asked for {args.width}x{args.height})"
                                              if (w, h) != (args.width, args.height) else ""))
    if w < 1280:
        print(f"  NOTE: {w}x{h} is low for 9x6 corners. Workable if the board fills the frame;")
        print("        a 1080p/720p webcam or a phone gives a better-conditioned calibration.")

    saved_poses, saved_files = [], []
    show_guide = True
    last_auto = 0.0
    countdown, countdown_start = None, 0.0
    message, message_until = "", time.time() + 6
    status = "READY"
    # cv2.imshow needs an interactive window station.  It fails outright on some setups, and because
    # this tool is built around a live preview that failure was total - the camera worked but not one
    # photo could be taken.  Headless mode keeps every quality gate and just reports to the console.
    headless = args.no_preview
    t_start = time.time()
    last_status = None

    if headless and args.warmup > 0:
        print(f"  [warmup] {args.warmup:.0f}s - position the board now, capture starts after this.")
        while time.time() - t_start < args.warmup:
            ok, frame = cap.read()
            if not ok:
                sys.exit("ERROR: camera stopped delivering frames during warmup.")
            time.sleep(0.03)
        print("  [warmup] done - capturing now.")

    while True:
        if headless and time.time() - t_start > args.max_seconds:
            print(f"\n  [stop] {args.max_seconds:.0f}s elapsed - stopping with {len(saved_files)} shot(s)")
            break
        ok, frame = cap.read()
        if not ok:
            print("\n  camera stopped delivering frames - quitting.")
            break

        raw = frame.copy()  # untouched camera frame; `frame` gets annotated for display and must
        # never be saved, because the guide shading and the corner markers drawn on it corrupt the
        # very features calibrate.py has to find and refine.

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = detect(g, pattern)

        area_frac = tilt = roll = sharp = 0.0
        pc = None
        if corners is not None:
            pc = pose(corners, w, h, pattern)
            area_frac, tilt, roll, cx, cy = pc
            sharp = sharpness(g, corners)
            cv2.drawChessboardCorners(frame, pattern, corners.astype(np.int32), True)

        n = len(saved_files)
        is_flat_shot = n < 3

        if corners is None:
            status, ok_to_shoot = "BOARD NOT FOUND  - show the whole board", False
        else:
            if area_frac < args.min_area:
                status, ok_to_shoot = "MOVE CLOSER", False
            elif sharp < args.min_sharp:
                status, ok_to_shoot = "HOLD STEADY / TAP TO FOCUS", False
            elif n == 0 and (tilt > args.flat_tilt or roll > args.flat_roll):
                status, ok_to_shoot = "LAY IT FLAT ON THE BELT (this shot defines the belt plane)", False
            elif is_flat_shot and (tilt > args.flat_tilt or roll > args.flat_roll):
                status, ok_to_shoot = "HOLD THE BOARD FLAT (square to the camera)", False
            elif not is_flat_shot and tilt < args.min_tilt and roll < args.min_roll:
                status, ok_to_shoot = "TILT OR ROLL THE BOARD (any new angle)", False
            elif too_similar(pc, saved_poses):
                status, ok_to_shoot = "NEW ANGLE NEEDED (this one is taken)", False
            else:
                status, ok_to_shoot = "LOOKING GOOD", True

        if headless:
            # Only report on a change, or a held-pose status would scroll a thousand lines.
            line = f"{len(saved_files):2d}/{args.target}  {status}"
            if status != last_status:
                if corners is not None:
                    line += (f"   [area {area_frac * 100:.0f}%  tilt {tilt:.1f}  "
                             f"roll {roll:.0f}  sharp {sharp:.0f}]")
                print("  " + line)
                last_status = status

        now = time.time()
        if ok_to_shoot and now - last_auto > args.cooldown:
            if countdown is None:
                countdown, countdown_start = 3, now
        else:
            countdown = None

        # ---- overlay -----------------------------------------------------------
        if show_guide:
            x0, y0, bw, bh = board_guide(w, h)
            shade = frame.copy()
            cv2.rectangle(shade, (0, 0), (w, y0), (0, 0, 0), -1)
            cv2.rectangle(shade, (0, y0 + bh), (w, h), (0, 0, 0), -1)
            cv2.rectangle(shade, (0, y0), (x0, y0 + bh), (0, 0, 0), -1)
            cv2.rectangle(shade, (x0 + bw, y0), (w, y0 + bh), (0, 0, 0), -1)
            frame = cv2.addWeighted(shade, 0.45, frame, 0.55, 0)
            col = (0, 230, 0) if ok_to_shoot else (0, 165, 255)
            cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + bh), col, 2)

        good_col = (0, 230, 0) if ok_to_shoot else (0, 140, 255)
        draw_text(frame, f"{n}/{args.target} saved", (14, 34), 0.85, (255, 255, 255), 2)
        draw_text(frame, status, (14, 68), 0.72, good_col, 2)
        # The first shot must show the board lying flat on the belt: that single view is what
        # calibrate.py takes the belt's homography from, so it is deliberately a different instruction
        # from the rest of the session rather than "the first of three similar flat shots".
        if n == 0:
            want = "1st shot: LAY THE BOARD FLAT ON THE BELT (saves as belt_00.png)"
            draw_text(frame, want, (14, h - 22), 0.55, (0, 230, 255), 1)
        elif is_flat_shot:
            want = f"{3 - n} flat shot(s) left - board square to the camera, vary the distance"
            draw_text(frame, want, (14, h - 22), 0.6, (220, 220, 220), 1)
        else:
            want = "now TILT or ROLL the board to a new angle each time"
            draw_text(frame, want, (14, h - 22), 0.6, (220, 220, 220), 1)
        if corners is not None:
            draw_text(frame, f"area {area_frac * 100:.0f}%   tilt {tilt:.1f}deg   roll {roll:.0f}deg   "
                             f"sharp {sharp:.0f}", (14, h - 52), 0.6, (220, 220, 220), 1)

        if countdown is not None:
            left = countdown - (now - countdown_start)
            cv2.circle(frame, (w // 2, h // 2), 46, (0, 0, 0), -1)
            draw_text(frame, str(int(np.ceil(left))), (w // 2 - 16, h // 2 + 20), 1.6, (0, 255, 255), 3)
            if left <= 0:
                cnt = int(np.count_nonzero([p.stem.startswith("board_") for p in out.glob("board_*.png")]))
                name = "belt_00.png" if n == 0 else f"board_{cnt:02d}.png"
                cv2.imwrite(str(out / name), raw)
                saved_poses.append(pc)
                saved_files.append(name)
                message = f"saved {name}"
                if headless:
                    print(f"  >>> SAVED {name}   ({len(saved_files)}/{args.target})")
                message_until = now + 2.5
                last_auto, countdown = now, None

        if message_until > now and message:
            draw_text(frame, message, (14, 100), 0.72, (0, 255, 0), 2)

        k = -1
        if not headless:
            cv2.imshow("Module B - checkerboard capture", frame)
            k = cv2.waitKey(1) & 0xFF

        if k in (KEY_Q, KEY_ESC):
            break
        if k == KEY_SPACEBAR and corners is not None:
            cnt = int(np.count_nonzero([p.stem.startswith("board_") for p in out.glob("board_*.png")]))
            name = "belt_00.png" if not saved_files else f"board_{cnt:02d}.png"
            cv2.imwrite(str(out / name), raw)
            saved_poses.append(pc)
            saved_files.append(name)
            message = f"saved {name} (forced)"
            message_until = time.time() + 2.5
        if k == ord("g"):
            show_guide = not show_guide
        if k == ord("d") and saved_files:
            gone = saved_files.pop()
            saved_poses.pop()
            (out / gone).unlink(missing_ok=True)
            message = f"deleted {gone}"
            message_until = time.time() + 2.0
        if k == ord("r") and saved_files:
            for f in saved_files:
                (out / f).unlink(missing_ok=True)
            saved_files, saved_poses = [], []
            message, message_until = "session reset", time.time() + 2.0

    cap.release()
    if not headless:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Some opencv-python builds ship without any highgui backend, in which case imshow
            # could never have worked either.  Nothing to tear down, so this is not worth failing on.
            pass

    print(BAR)
    print(f"  saved {len(saved_files)} shot(s) to {out}:")
    for f in saved_files:
        print(f"    {f}")
    print()
    if len(saved_files) < 10:
        print(f"  TOO FEW for a stable calibration - calibrate.py needs >= 10 detected boards.")
        print(f"  You have {len(saved_files)}. Run this again and take the rest.")
    else:
        print("  Good. Now calibrate:")
        print(f'    python module_B_calibration/calibrate.py --images "{out}" --pattern {cols}x{rows} \\')
        print(f"        --square-mm 25 --belt-image belt_00.png")
        print()
        print("  Then measure a box:")
        print("    python module_B_calibration/measure_box.py --ref-px 357 218 --ref-cm 30 20 --unk-px 200 120")
        print()
        print("  And run the pipeline with real centimetres:")
        print("    python pipeline.py --video data/conveyor.mp4 --box-height-cm H")
        print("    (H = the height of your box's top face above the belt, in cm)")
    print(BAR)


if __name__ == "__main__":
    main()