# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
import re
from typing import Any

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Plots the color schemes in `src/colorschemes.rs` for easy visualization.

Example:

    > {SCRIPT_PATH.name} --outfile colorscheme.png src/colorschemes.rs
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
        "image": {
            "cmap": "binary",
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


# {{{ main

Array = np.ndarray[Any, np.dtype[Any]]
PALETTE_NAME_RE = re.compile(r"const (\w+):.*")
RGB_RE = re.compile(r"\s*Rgb\(\[(\d+),\s*(\d+),\s*(\d+)\]\).*")


def main(
    filename: pathlib.Path,
    *,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    try:
        import matplotlib.pyplot as mp

        set_recommended_matplotlib()
    except ImportError:
        log.error("'matplotlib' package not found.")
        return 1

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    if outfile is None:
        outfile = "colorschemes"

    colorschemes = {}
    with open(filename, encoding="utf-8") as inf:
        try:
            while True:
                line = next(inf).strip()

                if not (match := PALETTE_NAME_RE.match(line)):
                    continue

                name = match.group(1)
                log.info("Found colorscheme '%s'.", name)

                colors = []
                while True:
                    line = next(inf).strip()
                    if not (match := RGB_RE.match(line)):
                        break

                    rgb = [int(d) for d in match.groups()]
                    colors.append(rgb)

                log.info("Found %d colors in colorscheme '%s'", len(colors), name)
                colorschemes[name] = np.array(colors)
        except StopIteration:
            pass

    from matplotlib.patches import Rectangle

    nrows = 4
    ncols = 8
    width = 1
    height = 1
    x = np.arange(0, ncols, width + 0.1)
    y = np.arange(0, nrows, height + 0.1)

    for name, colors in colorschemes.items():
        assert colors.shape == (nrows * ncols, 3)

        suffix = name.split("_")[-1].lower()
        outfilename = outfile.with_stem(f"{outfile.stem}-{suffix}")

        fig = mp.figure()
        ax = fig.gca()

        for i in range(nrows):
            for j in range(ncols):
                n = i * ncols + j
                sq = Rectangle(
                    (x[j], y[i]), width, height, fill=True, color=colors[n] / 256
                )
                ax.text(x[j] + 0.1, y[i] + 0.1, f"{n}", fontsize=24)
                ax.add_patch(sq)

        ax.relim()
        ax.autoscale_view()
        ax.set_axis_off()

        fig.savefig(outfilename)
        log.info("Saved colorscheme to '%s'.", outfilename)
        mp.close(fig)


# }}}


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
        default="colorschemes",
        help="Basename for output files",
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
            args.filename,
            outfile=args.outfile,
            overwrite=args.overwrite,
        )
    )
