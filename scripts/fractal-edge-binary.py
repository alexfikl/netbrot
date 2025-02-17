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


def main(
    filename: pathlib.Path,
    outfile: pathlib.Path | None = None,
    *,
    overwrite: bool = False,
) -> int:
    try:
        import cv2
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = filename.with_suffix(".bin")

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    # turn to grayscale
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    log.info("Loaded image '%s' of size %s.", filename, img.shape)

    # really clip the values
    gray[gray > 10] = 255
    gray[gray <= 10] = 0

    # edge detect
    edges = gray
    edges = cv2.Canny(gray, 100, 200)
    edges = edges.astype(bool).astype(np.uint8)

    np.savetxt(outfile, edges, fmt="%d", delimiter=" ")
    log.info("Saved binary file: '%s'", outfile)

    return 0


if __name__ == "__main__":
    import argparse

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(formatter_class=HelpFormatter)
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
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

    raise SystemExit(
        main(
            args.filename,
            args.outfile,
            overwrite=args.overwrite,
        )
    )
