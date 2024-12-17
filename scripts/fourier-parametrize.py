# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

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
This script extracts a Fourier description of the boundary of a given fractal
image generate by the main netbrot program. As expected, this is not going to
catch much in the way of the fractal nature of the object, but it does give a
reasonable approximation.

We use the standard edge detection to get the countour and apply the
Ramer-Douglas-Peucker algorithm to approximate the contour with a smaller number
of line segments. We then apply a standard FFT to this approximation to obtain
the Fourier modes.

Example:

    > {SCRIPT_PATH.name} --bbox -1.0 1.0 -1.0 1.0 exhibit-render.png
"""

# {{{ plotting settings


def set_recommended_matplotlib() -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (8, 8),
            "dpi": 300,
            "constrained_layout.use": True,
        },
        "text": {"usetex": True},
        "legend": {"fontsize": 20},
        "lines": {"linewidth": 2, "markersize": 5},
        "axes": {
            "labelsize": 28,
            "titlesize": 28,
            "grid": True,
            "grid.axis": "both",
            "grid.which": "both",
            # NOTE: preserve existing colors (the ones in "science" are ugly)
            "prop_cycle": mp.rcParams["axes.prop_cycle"],
        },
        "xtick": {"labelsize": 20, "direction": "inout"},
        "ytick": {"labelsize": 20, "direction": "inout"},
        "xtick.major": {"size": 6.5, "width": 1.5},
        "ytick.major": {"size": 6.5, "width": 1.5},
        "xtick.minor": {"size": 4.0},
        "ytick.minor": {"size": 4.0},
    }

    from contextlib import suppress

    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    for group, params in defaults.items():
        mp.rc(group, **params)


# }}}


def lerp(x: float, *, xfrom: tuple[float, float], xto: tuple[float, float]) -> float:
    a, b = xfrom
    t, s = xto

    return t + (x - a) / (b - a) * (s - t)


def parametrize_fourier(
    filenames: list[pathlib.Path],
    outfile: pathlib.Path | None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    nmodes: int | None = None,
    eps: float = 5.0e-4,
    overwrite: bool = False,
    debug: bool = False,
) -> int:
    try:
        import cv2
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    try:
        import matplotlib.pyplot as mp
    except ImportError:
        log.error("'matplotlib' package not found.")
        return 1

    if outfile is None:
        outfile = pathlib.Path(f"{SCRIPT_PATH.stem}-results.npz")

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    if bbox is None:
        bbox = (-1.0, 1.0, -1.0, 1.0)
    xmin, xmax, ymin, ymax = bbox

    set_recommended_matplotlib()

    results = []
    for filename in filenames:
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)
            return 1

        # read in the BGR image
        img = cv2.imread(filename)
        log.info("Loaded image '%s' of size %s.", filename, img.shape)

        # transform it to binary 0/255
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)

        # find the biggest contour
        contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        c = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(c, closed=True)
        log.info(
            "> Contour area %.5e perimeter %.5e points %d",
            cv2.contourArea(c),
            perimeter,
            len(c),
        )

        if debug:
            # draw contours
            output = img.copy()
            cv2.drawContours(output, [c], -1, (0, 0, 255), 3)
            cv2.imwrite(filename.with_stem(f"{filename.stem}-contour"), output)

        # approximate the contour
        approx = cv2.approxPolyDP(c, perimeter * eps, closed=True)
        log.info("> Approx tolerance %.5e points %d", perimeter * eps, len(approx))

        if debug:
            # draw approximated contour
            output = img.copy()
            cv2.drawContours(output, [approx], -1, (0, 0, 255), 3)
            cv2.imwrite(filename.with_stem(f"{filename.stem}-approx"), output)

        # get Fourier modes
        x = lerp(approx[:, 0, 0], xfrom=(0, img.shape[0]), xto=(xmin, xmax))
        y = lerp(approx[:, 0, 1], xfrom=(0, img.shape[1]), xto=(ymin, ymax))
        z = x + 1j * y
        zhat = np.fft.fft(z)

        # resample to desired number of modes
        if nmodes is None:
            zresampled = z
        else:
            k = np.fft.fftfreq(zhat.size, d=1.0 / zhat.size).reshape(-1, 1)
            theta = np.linspace(0.0, 2.0 * np.pi, nmodes)
            zresampled = np.einsum("i,ij->j", zhat, np.exp(1j * k * theta) / k.size)
            zhat = np.fft.fft(zresampled)

        # draw fourier modes
        if debug:
            k = np.fft.fftfreq(zhat.size, d=1.0 / zhat.size)

            fig = mp.figure()
            ax = fig.gca()

            ax.plot(k, zhat.real, "o-", label="Real")
            ax.plot(k, zhat.imag, "v-", label="Imag")
            ax.legend()

            fig.savefig(filename.with_stem(f"{filename.stem}-fourier"))
            mp.close(fig)

        # draw fourier contour
        if debug:
            fig = mp.figure()
            ax = fig.gca()

            if z.shape != zresampled.shape:
                ax.plot(z.real, z.imag, "o-", label="Original")
            ax.plot(zresampled.real, zresampled.imag, "o-", label="Fourier")

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.legend()

            fig.savefig(filename.with_stem(f"{filename.stem}-sample"))
            mp.close(fig)

        results.append(zhat)

    np.savez(outfile, modes=np.array(results, dtype=object))

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
    parser.add_argument("filenames", nargs="*", type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=(-1, 1, -1, 1),
        help="The bounding box in physical coordinates for the images",
    )
    parser.add_argument(
        "--modes",
        type=int,
        default=None,
        help="Number of Fourier modes to use for the parametrization",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="If true, save intermediate images for debugging",
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
        parametrize_fourier(
            args.filenames,
            args.outfile,
            bbox=args.bbox,
            nmodes=args.modes,
            overwrite=args.overwrite,
            debug=args.debug,
        )
    )
