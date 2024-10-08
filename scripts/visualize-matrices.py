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
This script plots out some matrices contained in an `.npz` file. It plots
* a description of the matrix structure as a grayscale image in log scale. This
  is mainly useful for sparse matrices with small entries.
* its eigenvalues. This can be used to see which matrices are close to singular.

Example:

    > {SCRIPT_PATH.name} --variable-name matrices data.npz
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


def main(
    filename: pathlib.Path,
    *,
    name: str = "matrices",
    overwrite: bool = False,
) -> int:
    if not filename.exists():
        log.error("File does not exist: '%s'", filename)
        return 1

    import matplotlib.pyplot as mp
    from matplotlib.colors import LogNorm

    data = np.load(filename, allow_pickle=True)
    matrices = data[name]

    set_recommended_matplotlib()

    for i in range(matrices.size):
        mat = matrices[i]
        eigs = np.linalg.eigvals(mat)
        kappa = np.linalg.cond(mat)

        fig, (ax1, ax2) = mp.subplots(1, 2)

        ax1.imshow(mat, norm=LogNorm() if mat.shape[0] > 32 else None)
        ax2.plot(eigs.real, eigs.imag, "o")
        ax2.set_xlim([-1.0, 1.0])
        ax2.set_title(rf"$\kappa = {kappa:.5e}$")

        outfile = filename.parent / f"{filename.stem}_{i:02d}"
        fig.savefig(outfile)
        mp.close(fig)

        log.info("Saving matrix %d to file '%s'.", i, outfile)

    return 0


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
        "-n",
        "--variable-name",
        default="matrices",
        help="Name of the variable containing matrices",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(
        main(
            args.filename,
            name=args.variable_name,
            overwrite=args.overwrite,
        )
    )
