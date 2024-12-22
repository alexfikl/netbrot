# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from contextlib import contextmanager
from typing import Any

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = ""

Array = np.ndarray[Any, np.dtype[Any]]


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


DEFAULT_EXTENT = (-10.25, 4.25, -6.0, 6.0)


@contextmanager
def figure(filename: pathlib.Path, *, overwrite: bool = False):
    import matplotlib.pyplot as mp

    if not overwrite and filename.exists():
        log.error("Output file exists (use --overwrite): '%s'.", filename)
        raise SystemExit(1)

    fig = mp.figure()

    try:
        yield fig
    finally:
        fig.savefig(filename)
        log.info("Saving figure: '%s'.", filename)
        mp.close(fig)


def image_mean_and_std(
    images: list[Array],
    *,
    outfile: pathlib.Path,
    extent: tuple[float, float, float, float],
    overwrite: bool = False,
) -> None:
    img_avg = 0
    img_var = 0

    for n, img in enumerate(images):
        img_var = n * (img_var + (img_avg - img) ** 2 / (n + 1)) / (n + 1)
        img_avg = (n * img_avg + img) / (n + 1)

    from matplotlib.colors import LogNorm

    # average
    with figure(outfile.with_stem(f"{outfile.stem}-avg"), overwrite=overwrite) as fig:
        ax = fig.gca()
        im = ax.imshow(img_avg, cmap="binary", extent=extent, norm=LogNorm())
        fig.colorbar(im, ax=ax)

    # standard deviation
    with figure(outfile.with_stem(f"{outfile.stem}-std"), overwrite=overwrite) as fig:
        ax = fig.gca()
        im = ax.imshow(np.sqrt(img_var), cmap="binary", extent=extent, norm=LogNorm())
        fig.colorbar(im, ax=ax)


def image_mean_squared_error(
    images: list[Array],
    *,
    outfile: pathlib.Path,
    overwrite: bool = False,
) -> None:
    result = np.array([
        [np.sum((imga - imgb) ** 2) / imga.size for imgb in images] for imga in images
    ])

    with figure(outfile.with_stem(f"{outfile.stem}-mse"), overwrite=overwrite) as fig:
        ax = fig.gca()
        im = ax.imshow(result)
        fig.colorbar(im, ax=ax)


def main(
    filenames: list[pathlib.Path],
    outfile: pathlib.Path | None = None,
    *,
    mode: str = "avg",
    extent: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
) -> int:
    try:
        import matplotlib.pyplot as mp

        set_recommended_matplotlib()
    except ImportError:
        log.error("'matplotlib' package not found.")
        return 1

    try:
        import cv2
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    if not filenames:
        log.warning("No input files given.")
        return 0

    if extent is None:
        extent = DEFAULT_EXTENT

    if outfile is None:
        ext = mp.rcParams["savefig.format"]
        outfile = pathlib.Path(f"result.{ext}")

    images = []
    for filename in filenames:
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)
            return 1

        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        log.info("Loaded image '%s' of size %s.", filename, img.shape)

        # convert to binary matrix
        if mode == "avg":
            img[img <= 10] = 1
            img[img > 10] = 0

        if images:
            prev = images[-1]
            if img.shape != prev.shape:
                log.error(
                    "Expected size %dx%d but image '%s' has size %dx%d.",
                    *prev.shape,
                    filename,
                    *img.shape,
                )
                return 1

        images.append(img)

    if mode == "avg":
        image_mean_and_std(images, outfile=outfile, extent=extent, overwrite=overwrite)
    elif mode == "mse":
        image_mean_squared_error(images, outfile=outfile, overwrite=overwrite)
    else:
        raise ValueError(f"Unknown mode: '{mode}'")

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
    parser.add_argument("filenames", nargs="+", type=pathlib.Path)
    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        default=DEFAULT_EXTENT,
        help="Extent limits (left, right, bottom, top) passed to imshow",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=("avg", "mse"),
        default="avg",
        help="Image comparison mode",
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

    raise SystemExit(
        main(
            args.filenames,
            args.outfile,
            mode=args.mode,
            extent=args.extent,
            overwrite=args.overwrite,
        )
    )
