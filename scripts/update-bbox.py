# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
This script updates the bounding box for a given exhibit based on its rendered
images. The filenames can contain a format string like `%02d` that will be
replaced by an index until there are no more files.

Example:

    > {SCRIPT_PATH.name} --pad 0.1 exhibit.json exhibit-render.png
"""


Array = np.ndarray[Any, np.dtype[Any]]


def lerp(x: float, *, xfrom: tuple[float, float], xto: tuple[float, float]) -> float:
    a, b = xfrom
    t, s = xto

    return t + (x - a) / (b - a) * (s - t)


def find_bounding_box(
    img: Array,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    pad: float = 0.0,
) -> tuple[float, float, float, float]:
    import cv2

    # get pixel bounding box
    coords = cv2.findNonZero(img)
    xnew, _, wnew, _ = cv2.boundingRect(coords)
    ynew = img.shape[1] // 2 - wnew // 2
    hnew = wnew

    # transform to physical coordinates
    xp = lerp(xnew, xfrom=(0, img.shape[0]), xto=(x, x + w))
    wp = lerp(xnew + wnew, xfrom=(0, img.shape[0]), xto=(x, x + w)) - xp

    padx = pad * wp
    xmin, xmax = xp - padx, xp + wp + padx
    ylen = (xmax - xmin) / 2.0

    return xmin, xmax, -ylen, ylen, (xnew, ynew), (xnew + wnew, ynew + hnew)


def update_bbox(
    jsonfile: pathlib.Path,
    pngfile: pathlib.Path,
    outfile: pathlib.Path | None = None,
    *,
    pad: float = 0.1,
    overwrite: bool = False,
) -> int:
    try:
        import cv2
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    if not jsonfile.exists():
        log.error("File does not exist: '%s'.", jsonfile)
        return 1

    if not pngfile.exists():
        log.error("File does not exist: '%s'.", pngfile)
        return 1

    if outfile is None:
        outfile = jsonfile

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    # {{{ determine bounding box

    with open(jsonfile, encoding="utf-8") as inf:
        data = json.load(inf)
    x = data["upper_left"][0]
    y = data["lower_right"][1]
    w = data["lower_right"][0] - x
    h = data["upper_left"][1] - y

    img = cv2.imread(pngfile, cv2.IMREAD_GRAYSCALE)
    log.info("Loaded image '%s' of size %s.", pngfile, img.shape)
    log.info("Bounding box: [%.6f, %.6f] x [%.6f, %.6f]", x, x + w, y, y + h)

    img[img > 10] = 255
    img[img <= 10] = 0

    xmin, xmax, ymin, ymax, _, _ = find_bounding_box(255 - img, x, y, w, h, pad=pad)
    log.info("              [%.6f, %.6f] x [%.6f, %.6f]", xmin, xmax, ymin, ymax)

    if xmin < x or xmax > x + w or ymin < y or ymax > y + h:
        log.error("Failed to find bounding box.")
        return 1

    # cv2.rectangle(img, pt0, pt1, 128, 2)
    # cv2.imwrite(pngfile.with_stem(f"{pngfile.stem}-bbox"), img)

    # }}}

    # {{{ save data

    data["upper_left"] = [xmin, ymax]
    data["lower_right"] = [xmax, ymin]

    with open(outfile, "w", encoding="utf-8") as outf:
        json.dump(data, outf, indent=2, sort_keys=False)

    # }}}

    return 0


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description=SCRIPT_LONG_HELP,
    )
    parser.add_argument("filenames", nargs=2, type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.1,
        help="Percentage (in [0, 1]) of width to pad the bounding boxes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    if not 0 <= args.pad <= 1:
        log.error("Padding not in [0, 1]: %g", args.pad)
        raise SystemExit(1)

    n = 0
    ret = 0
    while True:
        # get filenames
        jsonfile = args.filenames[0]
        jsonfile = jsonfile.with_stem(jsonfile.stem % (n,))
        pngfile = args.filenames[1]
        pngfile = pngfile.with_stem(pngfile.stem % (n,))
        outfile = args.outfile
        if outfile is not None:
            outfile = outfile.with_stem(outfile.stem % (n,))

        if not jsonfile.exists():
            break

        ret += update_bbox(
            jsonfile,
            pngfile,
            outfile=outfile,
            pad=args.pad,
            overwrite=args.overwrite,
        )

        n += 1

    raise SystemExit(int(ret == 0))
