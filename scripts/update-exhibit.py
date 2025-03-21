# SPDX-FileCopyrightText: 2025 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
This file updates a given exhibit JSON file to have the proper format. In particular,
it updates the escape radius and the bounding box.

Example:

    > {SCRIPT_PATH.name} --image exhibit.png --pad 0.1 exhibit.json
"""


Array = np.ndarray[tuple[int, ...], np.dtype[np.floating]]


def estimate_escape_radius(mat: Array) -> float:
    n = mat.shape[0]
    sigma = np.linalg.svdvals(mat)

    return float(2.0 * np.sqrt(n) / np.min(sigma) ** 2)


def lerp(x: float, *, xfrom: tuple[float, float], xto: tuple[float, float]) -> float:
    a, b = xfrom
    t, s = xto

    return t + (x - a) / (b - a) * (s - t)


def find_bounding_box(
    imagefile: pathlib.Path,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    pad: float = 0.0,
) -> tuple[float, float, float, float]:
    import cv2

    img = cv2.imread(imagefile)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log.info("Loaded image '%s' of size %s.", imagefile, gray.shape)

    blue = img[:, :, 0]
    gray[blue > 225] = 255
    gray[blue <= 225] = 0
    gray = 255 - gray

    # get pixel bounding box
    coords = cv2.findNonZero(gray)
    xnew, ynew, wnew, hnew = cv2.boundingRect(coords)
    log.info("%d %d x %d %d", xnew, xnew + wnew, ynew, ynew + hnew)

    # transform to physical coordinates
    xp = lerp(xnew, xfrom=(0, gray.shape[1]), xto=(x, x + w))
    wp = lerp(xnew + wnew, xfrom=(0, gray.shape[1]), xto=(x, x + w)) - xp
    yp = lerp(ynew, xfrom=(0, gray.shape[0]), xto=(y, y + h))
    hp = lerp(ynew + hnew, xfrom=(0, gray.shape[0]), xto=(y, y + h)) - yp

    padx = pad * wp
    xmin, xmax = xp - padx, xp + wp + padx

    pady = pad * hp
    ymin, ymax = yp - pady, -yp + pady

    return xmin, xmax, ymin, ymax


def main(
    jsonfile: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    imagefile: pathlib.Path | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    pad: float = 0.0,
    overwrite: bool = False,
) -> int:
    try:
        import cv2  # noqa: F401
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    if not jsonfile.exists():
        log.error("Exhibit file does not exist: '%s'.", jsonfile)
        return 1

    if outfile is None:
        outfile = jsonfile

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --force): '%s'.", outfile)
        return 1

    if imagefile is not None and not imagefile.exists():
        log.error("Provided image file does not exist: '%s'.", imagefile)
        return 1

    if bbox is not None and (bbox[0] >= bbox[1] or bbox[2] >= bbox[3]):
        log.error("Invalid bbox (non-positive size): %s", bbox)
        return 1

    if imagefile is not None and bbox is not None:
        log.error("Cannot provide both '--bbox' and '--image'.")
        return 1

    if pad < 0:
        log.error("Padding cannot be negative: %g", pad)
        return 1

    # {{{ update exhibit

    import json

    with open(jsonfile, encoding="utf-8") as inf:
        data = json.load(inf)

    # escape radius
    elements, *shape = data["mat"]
    mat = np.array([e_r for e_r, _ in elements]).reshape(*shape).T
    escape_radius = estimate_escape_radius(mat)
    log.info("[NEW] radius: %.12e -> %.12e", data["escape_radius"], escape_radius)

    data["escape_radius"] = escape_radius

    # bounding box
    x = data["upper_left"][0]
    y = data["lower_right"][1]
    w = data["lower_right"][0] - x
    h = data["upper_left"][1] - y

    if imagefile is not None:
        bbox = find_bounding_box(imagefile, x, y, w, h, pad=pad)

    if bbox is not None:
        log.info(
            "[NEW] bbox: [%.6f, %.6f]x[%.6f, %.6f] -> [%.6f, %.6f]x[%.6f, %.6f]",
            x,
            x + w,
            y,
            y + h,
            bbox[0],
            bbox[1],
            bbox[2],
            bbox[3],
        )
        data["upper_left"] = [bbox[0], bbox[3]]
        data["lower_right"] = [bbox[1], bbox[2]]

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
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Name of the output file (the input is overwritten if not provided)",
    )
    parser.add_argument(
        "--image",
        type=pathlib.Path,
        default=None,
        help="A rendered image of the exibit used to determine the bounding box",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        help="The bounding box in physical coordinates for the images",
    )
    parser.add_argument(
        "--pad",
        type=float,
        default=0.0,
        help="Padding added when using an image to determine the bounding box",
    )
    parser.add_argument(
        "-f",
        "--force",
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

    raise SystemExit(
        main(
            args.filename,
            outfile=args.outfile,
            imagefile=args.image,
            bbox=args.bbox,
            pad=args.pad,
            overwrite=args.force,
        )
    )
