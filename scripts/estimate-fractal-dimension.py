# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())


def main(filename: pathlib.Path) -> int:
    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    try:
        import cv2
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    # turn to grayscale
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log.info("Loaded image '%s' of size %s.", filename, img.shape)

    pixels = np.array(np.nonzero(gray)).T
    scales = np.logspace(0.01, 1, num=10, endpoint=False, base=2)
    ns = np.empty_like(scales)

    for i in range(scales.size):
        xbins = np.arange(0, img.shape[1], scales[i])
        ybins = np.arange(0, img.shape[0], scales[i])
        H, _ = np.histogramdd(pixels, bins=(xbins, ybins))
        ns[i] = np.sum(H > 0)

    coeffs = np.polyfit(np.log(scales), np.log(ns), 1)
    log.info("Estimated dimension: %.5f", -coeffs[0])

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(main(args.filename))
