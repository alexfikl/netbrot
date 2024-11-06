# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any, TypedDict

import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
This script generates a collection of JSON files that can be used with the main
`netbrot` executable to define the fractal that will be rendered. The JSON
files have a very specific format (so that they can be loaded by `serde`), so
be careful when playing with them manually!

This script generates some random NxN matrices of a desired type. Currently
this supports: `fixed` for some matrices we like, `feedforward` for lower
triangular matrices, and `equalrow` for matrices with equal row sums.

The bounding box for rendering can be specified using the `--xlim` and `--ylim`
parameters. The resulting JSON files has the following fields
```JSON
{{
    "mat": [[[mij.real, mij.imag], ...], nx, ny],
    "escape_radius": 1.0,
    "upper_left": [ux, uy],
    "lower_right": [lx, ly],
}}
```

Example:

    > {SCRIPT_PATH.name} --type feedforward --size 10 --count 32
"""

# {{{ utils

Array = np.ndarray[Any, np.dtype[Any]]

DEFAULT_UPPER_LEFT = (-1.0, 1.0)
DEFAULT_LOWER_RIGHT = (1.0, -1.0)


def serde_matrix_format(mat: Array) -> list[Any]:
    result = [[float(item), 0.0] for row in mat.T for item in row]
    return [result, *mat.shape]


def estimate_escape_radius(mat: Array) -> float:
    n = mat.shape[0]
    sigma = np.linalg.svdvals(mat)

    return 2.0 * np.sqrt(n) / np.min(sigma) ** 2


def dump(
    outfile: pathlib.Path,
    mat: Array,
    upper_left: tuple[float, float] = DEFAULT_UPPER_LEFT,
    lower_right: tuple[float, float] = DEFAULT_LOWER_RIGHT,
    *,
    max_escape_radius: float | None = None,
    overwrite: bool = False,
) -> int:
    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    if max_escape_radius is None:
        max_escape_radius = np.inf

    with open(outfile, "w", encoding="utf-8") as outf:
        escape_radius = estimate_escape_radius(mat)
        exhibit_escape_radius = min(escape_radius, max_escape_radius)
        log.info(
            "Dumping exhibit '%s': shape %s (cond %.3e) escape radius %g (real %g)",
            outfile.stem,
            mat.shape,
            np.linalg.cond(mat),
            exhibit_escape_radius,
            escape_radius,
        )

        json.dump(
            {
                "mat": serde_matrix_format(mat),
                "escape_radius": exhibit_escape_radius,
                "upper_left": upper_left,
                "lower_right": lower_right,
            },
            outf,
            indent=2,
            sort_keys=False,
        )

    log.info("Saved matrix in '%s'.", outfile)
    return 0


# }}}


# {{{ random


class Exhibit(TypedDict):
    mat: Array
    upper_left: tuple[float, float]
    lower_right: tuple[float, float]


def generate_fixed_matrix() -> list[Exhibit]:
    return [
        Exhibit(
            mat=np.array([[1.0, 0.8], [1.0, -0.5]]),
            upper_left=(-0.9, 0.6),
            lower_right=(0.4, -0.6),
        ),
        Exhibit(
            mat=np.array([[1.0, 1.0], [0.0, 1.0]]),
            upper_left=(-0.9, 0.6),
            lower_right=(0.4, -0.6),
        ),
        Exhibit(
            mat=np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -1.0]]),
            upper_left=(-1.25, 0.75),
            lower_right=(0.5, -0.75),
        ),
        Exhibit(
            mat=np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -1.0]]),
            upper_left=(-1.025, 0.025),
            lower_right=(-0.975, -0.025),
        ),
    ]


def generate_feed_forward(
    rng: np.random.Generator,
    matrix_size: int,
    exhibit_count: int,
    *,
    upper_left: tuple[float, float] = DEFAULT_UPPER_LEFT,
    lower_right: tuple[float, float] = DEFAULT_LOWER_RIGHT,
    parametric: bool = False,
) -> list[Exhibit]:
    matrices = []

    if parametric:
        omega = np.linspace(0.5, 1.0, exhibit_count)
        rho = np.linspace(-3, 5, exhibit_count)
        tau = omega * (rho + 1)

        for n in range(exhibit_count):
            matrices.append(
                np.array([
                    [omega[n], 0.0],
                    [2 * omega[n] - tau[n], tau[n] - omega[n]],
                ])
            )
    else:
        triu = np.triu_indices(matrix_size, k=1)
        for _ in range(exhibit_count):
            mat = rng.uniform(size=(matrix_size, matrix_size))
            mat[triu] = 0.0
            matrices.append(mat)

    return [
        Exhibit(mat=mat, upper_left=upper_left, lower_right=lower_right)
        for mat in matrices
    ]


def generate_equal_row(
    rng: np.random.Generator,
    matrix_size: int,
    exhibit_count: int,
    *,
    upper_left: tuple[float, float] = DEFAULT_UPPER_LEFT,
    lower_right: tuple[float, float] = DEFAULT_LOWER_RIGHT,
    parametric: bool = False,
) -> list[Exhibit]:
    matrices = []

    if parametric:
        omega = np.linspace(0.55, 1.0, exhibit_count)
        for n in range(exhibit_count):
            matrices.append(
                np.array([
                    [omega[n] / 2, omega[n] / 2],
                    [1.0, omega[n] - 1.0],
                ])
            )
    else:
        for _ in range(exhibit_count):
            mat = rng.uniform(size=(matrix_size, matrix_size))

            rows = np.sum(mat, axis=1)
            mat *= rows[0] / rows.reshape(-1, 1)
            assert np.all(np.isclose(np.sum(mat, axis=1), rows[0]))

            matrices.append(mat)

    return [
        Exhibit(mat=mat, upper_left=upper_left, lower_right=lower_right)
        for mat in matrices
    ]


def generate_random_matrix(
    matrix_size: int,
    nmatrices: int,
    *,
    mat_type: str = "fixed",
    upper_left: tuple[float, float] | None = None,
    lower_right: tuple[float, float] | None = None,
    max_escape_radius: float | None = None,
    parametric: bool = False,
    rng: np.random.Generator | None = None,
    outfile: pathlib.Path | None = None,
    overwrite: bool = False,
) -> int:
    if rng is None:
        rng = np.random.default_rng(seed=42)

    if outfile is None:
        outfile = pathlib.Path(f"exhibit-random-{matrix_size}x{matrix_size}.json")

    if upper_left is None:
        upper_left = DEFAULT_UPPER_LEFT

    if lower_right is None:
        lower_right = DEFAULT_LOWER_RIGHT

    if upper_left[0] > lower_right[0]:
        log.error("Invalid bounds: xmin %s xmax %s", upper_left[0], lower_right[0])
        return 1

    if upper_left[1] < lower_right[1]:
        log.error("Invalid bounds: ymin %s ymax %s", upper_left[1], lower_right[1])
        return 1

    if mat_type == "fixed":
        exhibits = generate_fixed_matrix()
    elif mat_type == "feedforward":
        exhibits = generate_feed_forward(
            rng,
            matrix_size,
            nmatrices,
            upper_left=upper_left,
            lower_right=lower_right,
            parametric=parametric,
        )
    elif mat_type == "equalrow":
        exhibits = generate_equal_row(
            rng,
            matrix_size,
            nmatrices,
            upper_left=upper_left,
            lower_right=lower_right,
            parametric=parametric,
        )
    else:
        raise ValueError(f"Unknown matrix type: '{mat_type}'")

    ret = 0
    width = len(str(len(exhibits)))
    for i, ex in enumerate(exhibits):
        outfile_i = outfile.with_stem(f"{outfile.stem}-{i:0{width}}")
        ret |= dump(
            outfile_i,
            **ex,
            max_escape_radius=max_escape_radius,
            overwrite=overwrite,
        )

    return ret


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
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        default=None,
        help="Basename for output files (named '{basename}-XX.json')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "-x",
        "--xlim",
        type=float,
        nargs=2,
        default=(DEFAULT_UPPER_LEFT[0], DEFAULT_LOWER_RIGHT[0]),
        help="Rendering bounds (in physical space) for the x coordinate",
    )
    parser.add_argument(
        "-y",
        "--ylim",
        type=float,
        nargs=2,
        default=(DEFAULT_LOWER_RIGHT[1], DEFAULT_UPPER_LEFT[1]),
        help="Rendering bounds (in physical space) for the y coordinate",
    )
    parser.add_argument(
        "-r",
        "--escape-radius",
        default=None,
        type=float,
        help="Maximum escape radius",
    )
    parser.add_argument(
        "-t",
        "--type",
        choices=("fixed", "feedforward", "equalrow"),
        default="fixed",
    )
    parser.add_argument(
        "-p",
        "--parametric",
        action="store_true",
        help="Generate simple 2x2 matrices with from a parameter",
    )
    parser.add_argument(
        "-n",
        "--size",
        default=2,
        type=int,
        help="Size of the matrix in exhibits (ignored for --parametric)",
    )
    parser.add_argument(
        "-m",
        "--count",
        default=10,
        type=int,
        help="Number of exhibits to generate",
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
        generate_random_matrix(
            args.size,
            args.count,
            mat_type=args.type,
            upper_left=(args.xlim[0], args.ylim[1]),
            lower_right=(args.xlim[1], args.ylim[0]),
            max_escape_radius=args.escape_radius,
            parametric=args.parametric,
            outfile=args.outfile,
            overwrite=args.overwrite,
        )
    )
