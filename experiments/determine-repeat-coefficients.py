# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import rich.logging
import sympy as sp

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

Array = np.ndarray[Any, np.dtype[Any]]
Matrix = Array

TEMPLATE = """\
"""


def netbrot(z: Array, mat: Matrix, c: complex) -> Array:
    return (mat @ z) ** 2 + c


def find_coefficients(eq: sp.Poly, z: Array) -> dict[tuple[int, ...], sp.Expr]:
    return {
        pows: eq.coeff_monomial(sp.prod([z_i**n_i for z_i, n_i in zip(z, pows)]))
        for pows in eq.monoms()
    }


def main(filename: pathlib.Path, *, nperiod: int = 1) -> int:
    # {{{ read in matrix

    import json

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    with open(filename, encoding="utf-8") as inf:
        data = json.load(inf)

    elements, *shape = data["mat"]
    ndim = shape[1]
    assert shape[0] == shape[1]

    # }}}

    # {{{ set up symbolics

    c = sp.Symbol("c")
    z = np.array(sp.symbols([f"z_{i}" for i in range(ndim)]))
    mat = np.array([sp.Rational(str(e_r)) for e_r, _ in elements]).reshape(ndim, ndim).T
    mat = np.array(
        sp.symbols([f"A_{i}_{j}" for i in range(ndim) for j in range(ndim)])
    ).reshape(ndim, ndim)

    result = netbrot(z, mat, c)
    for _ in range(1, nperiod):
        result = netbrot(result, mat, c)

    equations = [sp.Poly(sp.expand(f_i), *z) for f_i in result - z]
    coefficients = [find_coefficients(eq, z) for eq in equations]

    for i in range(ndim):
        log.info("Equation: %s", equations[i])
        log.info("Coefficients: %s", coefficients[i])

    # }}}

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument("-n", default=1, type=int, help="Number of self-compositions")
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only show error messages",
    )
    args = parser.parse_args()

    if not args.quiet:
        log.setLevel(logging.INFO)

    raise SystemExit(main(args.filename, nperiod=args.n))
