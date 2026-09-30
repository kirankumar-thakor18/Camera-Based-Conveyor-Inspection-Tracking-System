"""
Print an EXACTLY-scaled checkerboard for Module B's calibration, plus a capture helper.

The assignment asks for 10-15 photos of a checkerboard taken from different angles, then
cv2.calibrateCamera to recover K and the radial distortion coefficients.  The board's square size has to
be exact in centimetres, otherwise every recovered focal length - and therefore every cm measurement in
the whole project - is wrong.

Produces:
  data/checkerboard/checkerboard_9x6.pdf   print at 100 % scale (no "fit to page"!)
  data/checkerboard/checkerboard_9x6.png   same board as an image, for a tablet/laptop screen
  data/checkerboard/PRINT_AND_SHOOT.md     the shot list + how to verify the print scale

Run:  python tools/make_checkerboard.py
      python tools/make_checkerboard.py --cols 9 --rows 6 --square-mm 25
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "checkerboard"


def build(cols, rows, sq_mm):
    """Draw a board of (cols+1)x(rows+1) squares -> cols x rows INNER corners."""
    sq = sq_mm / 25.4  # matplotlib works in inches
    w, h = (cols + 1) * sq, (rows + 1) * sq
    # A quiet zone of at least 2 squares around the board: findChessboardCorners needs plain border on
    # all four sides or it cannot lock onto the outer squares.
    quiet = 2 * sq
    page_w, page_h = w + 2 * quiet, h + 2 * quiet
    fig = plt.figure(figsize=(page_w, page_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # checkerboard: (cols+1) x (rows+1) squares, origin at the quiet-zone corner
    x0, y0 = quiet, quiet + (page_h - h) / 2 - quiet + quiet
    x0, y0 = quiet, page_h - quiet - h
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                continue
            ax.add_patch(plt.Rectangle((x0 + c * sq, y0 + (rows - r) * sq), sq, sq,
                                       facecolor="black", edgecolor="none"))

    # 100 mm scale bar so the printed size can be checked with a real ruler after printing.
    bar = 100 / 25.4
    bx, by = x0, y0 - quiet * 0.45
    ax.add_patch(plt.Rectangle((bx, by), bar, sq * 0.35, facecolor="#c8102e", edgecolor="none"))
    ax.text(bx + bar + sq * 0.3, by + sq * 0.05, "100 mm exactly - measure this after printing",
            fontsize=7, va="center", color="#c8102e")
    ax.text(x0, y0 + h + quiet * 0.25,
            f"INNER corners = {cols} x {rows}   |   square = {sq_mm:g} mm   |   "
            f"board = {(cols + 1) * sq_mm:g} x {(rows + 1) * sq_mm:g} mm   |   "
            f"run calibrate.py --pattern {cols}x{rows} --square-mm {sq_mm:g}",
            fontsize=7, va="center")
    return fig, cols * rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cols", type=int, default=9, help="INNER corners across")
    ap.add_argument("--rows", type=int, default=6, help="INNER corners down")
    ap.add_argument("--square-mm", type=float, default=25.0)
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    if a.cols < 3 or a.rows < 3:
        raise SystemExit("need at least 3x3 inner corners")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, n_corners = build(a.cols, a.rows, a.square_mm)

    pdf = OUT / f"checkerboard_{a.cols}x{a.rows}.pdf"
    fig.savefig(pdf)  # vector PDF: scale-exact as long as it is printed at 100 %
    png = OUT / f"checkerboard_{a.cols}x{a.rows}.png"
    fig.savefig(png, dpi=a.dpi)
    plt.close(fig)

    # Sanity check on the PNG: a printed-then-rephotographed board is what calibrate.py sees, so verify
    # the generated image really contains the expected number of square edges.
    import cv2
    im = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
    dark = (im < 128).astype(np.uint8)
    row = dark[dark.shape[0] // 2]
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate(([0], row, [0]))) != 0))
    edges = int(np.count_nonzero(np.diff(np.concatenate(([0], row, [0]))) != 0))

    print(f"wrote {pdf.name}  ({a.cols + 1}x{a.rows + 1} squares, {a.square_mm:g} mm, "
          f"{(a.cols + 1) * a.square_mm:g} x {(a.rows + 1) * a.square_mm:g} mm board)")
    print(f"wrote {png.name}  ({im.shape[1]}x{im.shape[0]} px @ {a.dpi} dpi)")
    print(f"mid-row dark/light transitions in the PNG: {edges} "
          f"(a {a.cols + 1}-square row should give {a.cols + 1})")
    print(f"inner corners the calibrator will look for: {n_corners}")
    print()
    print("PRINTING:")
    print("  1. Open the PDF and print at 100 % / 'Actual size'. NOT 'fit to page' or 'shrink'.")
    print("  2. Measure the red bar with a ruler. If it is not 100 mm, adjust the print scale by")
    print("     100/measured and re-print, or edit --square-mm to match reality.")
    print("  3. Stick the board to something flat and rigid. Do not print on paper you can flex.")
    print(f"  4. Tape it to the belt/table for the '--belt-image' shot, and hold it at various")
    print("     angles for the rest - see PRINT_AND_SHOOT.md for the full shot list.")


if __name__ == "__main__":
    main()