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

Array = np.ndarray[Any, np.dtype[Any]]

# {{{ utils


def shift_minimum_singular_value(mat: Any, max_escape_radius: float = 20.0) -> Array:
    n = mat.shape[0]

    sigma_min = np.sqrt(2.0 * np.sqrt(n) / max_escape_radius)
    sigma = np.linalg.svdvals(mat)

    return mat * sigma_min / np.min(sigma)


# }}}


# {{{ generate_default


def generate_default(outfile: pathlib.Path, *, overwrite: bool = False) -> int:
    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    matrices = np.empty(4, dtype=object)
    matrices[0] = np.array([[1.0, 0.8], [1.0, -0.5]])
    matrices[1] = np.array([[1.0, 1.0], [0.0, 1.0]])
    matrices[2] = np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 1.0, -1.0]])
    matrices[3] = matrices[2]

    upper_lefts = np.array([
        complex(-0.9, 0.6),
        complex(-0.9, 0.6),
        complex(-1.25, 0.75),
        complex(-1.025, 0.025),
    ])
    lower_rights = np.array([
        complex(0.4, -0.6),
        complex(0.4, -0.6),
        complex(0.5, -0.75),
        complex(-0.975, -0.025),
    ])

    np.savez(
        outfile,
        matrices=matrices,
        upper_lefts=upper_lefts,
        lower_rights=lower_rights,
    )
    log.info("Saved results in '%s'.", outfile)

    return 0


# }}}

# {{{ generate_feed_forward


def generate_feed_forward(
    matrix_size: int,
    exhibit_count: int,
    outfile: pathlib.Path,
    *,
    overwrite: bool = False,
) -> int:
    rng = np.random.default_rng(seed=42)
    triu = np.triu_indices(matrix_size, k=1)

    matrices = np.empty(exhibit_count, dtype=object)
    for n in range(exhibit_count):
        mat = rng.uniform(size=(matrix_size, matrix_size))
        mat[triu] = 0.0
        matrices[n] = shift_minimum_singular_value(mat)

    # FIXME: can we determine these from the eigenvalues?
    upper_lefts = np.full(len(matrices), complex(-0.075, 0.04))
    lower_rights = np.full(len(matrices), complex(0.02, -0.04))
    np.savez(
        outfile,
        matrices=matrices,
        upper_lefts=upper_lefts,
        lower_rights=lower_rights,
    )

    log.info("Saved results in '%s'.", outfile)


# }}}


# {{{ generate_equal_row


def generate_equal_row(
    matrix_size: int,
    exhibit_count: int,
    outfile: pathlib.Path,
    *,
    parametric: bool = False,
    overwrite: bool = False,
) -> int:
    rng = np.random.default_rng(seed=42)

    matrices = np.empty(exhibit_count, dtype=object)
    if parametric:
        omega = np.linspace(0.5, 1.0, exhibit_count)
        for n in range(exhibit_count):
            matrices[n] = np.array([
                [omega[n] / 2, omega[n] / 2],
                [1.0, omega[n] - 1.0],
            ])
    else:
        for n in range(exhibit_count):
            mat = rng.uniform(size=(matrix_size, matrix_size))

            rows = np.sum(mat, axis=1)
            mat *= rows[0] / rows.reshape(-1, 1)
            assert np.all(np.isclose(np.sum(mat, axis=1), rows[0]))

            matrices[n] = shift_minimum_singular_value(mat)

    # FIXME: can we determine these from the eigenvalues?
    upper_lefts = np.full(len(matrices), complex(-1.25, 0.75))
    lower_rights = np.full(len(matrices), complex(0.5, -0.75))
    np.savez(
        outfile,
        matrices=matrices,
        upper_lefts=upper_lefts,
        lower_rights=lower_rights,
    )
    log.info("Saved results in '%s'.", outfile)


# }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "-n", "--size", default=2, type=int, help="Size of the matrix in exhibits"
    )
    parser.add_argument(
        "-m", "--count", default=10, type=int, help="Number of exhibits to generate"
    )
    subparsers = parser.add_subparsers()

    # default
    parser0 = subparsers.add_parser("default", help="Generate some default examples")
    parser0.set_defaults(
        func=lambda args: generate_default(
            args.outfile or "defaults.npz", overwrite=args.overwrite
        )
    )

    # feed forward matrices
    parsera = subparsers.add_parser(
        "feedfwd", help="Generate random feed forward matrices"
    )
    parsera.set_defaults(
        func=lambda args: generate_feed_forward(
            args.size,
            args.count,
            args.outfile or "feedfwd.npz",
            overwrite=args.overwrite,
        )
    )

    # equal row
    parserb = subparsers.add_parser(
        "equalrow", help="Generate random matrices that have equal row sums"
    )
    parserb.add_argument("-p", "--parametric", action="store_true")
    parserb.set_defaults(
        func=lambda args: generate_equal_row(
            args.size,
            args.count,
            args.outfile or "equalrow.npz",
            parametric=args.parametric,
            overwrite=args.overwrite,
        )
    )

    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(args.func(args))
