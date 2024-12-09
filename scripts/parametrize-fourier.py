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
Example:

    > {SCRIPT_PATH.name} exhibit-render.png
"""

# {{{ plotting settings


def set_recommended_matplotlib() -> None:
    try:
        import matplotlib.pyplot as mp
    except ImportError:
        return

    defaults: dict[str, dict[str, Any]] = {
        "figure": {
            "figsize": (16, 8),
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


def parametrize_fourier(
    filenames: list[pathlib.Path],
    outfile: pathlib.Path | None,
    *,
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

    if not overwrite and outfile is not None and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    set_recommended_matplotlib()

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
        x = approx[:, 0, 0] + 1j * approx[:, 0, 1]
        xhat = np.fft.fft(x)

        # resample to desired number of modes
        if nmodes is None:
            xresampled = x
        else:
            k = np.fft.fftfreq(xhat.size, d=1.0 / xhat.size).reshape(-1, 1)
            theta = np.linspace(0.0, 2.0 * np.pi, nmodes)
            xresampled = np.einsum("i,ij->j", xhat, np.exp(1j * k * theta) / k.size)
            xhat = np.fft.fft(xresampled)

        if debug:
            # draw fourier modes
            k = np.fft.fftfreq(xhat.size, d=1.0 / xhat.size)

            fig = mp.figure()
            ax = fig.gca()

            ax.plot(k, xhat.real, "o-", label="Real")
            ax.plot(k, xhat.imag, "v-", label="Imag")
            ax.set_xlim([-32, 32])
            ax.legend()

            fig.savefig(filename.with_stem(f"{filename.stem}-fourier"))
            mp.close(fig)

        if debug:
            # draw fourier contour
            fig = mp.figure()
            ax = fig.gca()

            ax.plot(x.real, x.imag, "o-", label="Original")
            ax.plot(xresampled.real, xresampled.imag, "o-", label="Fourier")
            ax.set_aspect("equal")
            ax.legend()

            fig.savefig(filename.with_stem(f"{filename.stem}-sample"))
            mp.close(fig)

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
            nmodes=args.modes,
            overwrite=args.overwrite,
            debug=args.debug,
        )
    )
