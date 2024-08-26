# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass
from typing import Any

import jinja2
import numpy as np
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

TEMPLATE = """\
// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// NOTE: This file has been generated by `scripts/generate-matrix-gallery.py`.
// DO NOT MODIFY it manually.

use nalgebra::{dmatrix, DMatrix};
use num::complex::Complex64;

macro_rules! c64 {
    ($re: literal) => {
        Complex64 { re: $re, im: 0.0 }
    };
}

pub struct Exhibit {
    /// Matrix used in the iteration.
    pub mat: DMatrix<Complex64>,
    /// Escape radius for this matrix.
    pub escape_radius: f64,
    /// Bounding box for the points.
    pub upper_left: Complex64,
    pub lower_right: Complex64,
}

#[allow(dead_code)]
#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ExhibitId {
(((- for ex in exhibits )))
    ((* ex.identifier *)),
(((- endfor )))
}

pub fn get_exhibit(id: ExhibitId) -> Exhibit {
    match id {
(((- for ex in exhibits )))
        ExhibitId::((* ex.identifier *)) => Exhibit {
            mat: dmatrix![
                ((* ex.stringified_mat | indent(width=16) *))
            ],
            escape_radius: ((* ex.escape_radius *)),
            upper_left: Complex64 {
                re: ((* ex.upper_left.real *)),
                im: ((* ex.upper_left.imag *)),
            },
            lower_right: Complex64 {
                re: ((* ex.lower_right.real *)),
                im: ((* ex.lower_right.imag *)),
            },
        },
(((- endfor )))
    }
}
"""


@dataclass(frozen=True)
class Exhibit:
    name: str
    """Name of the exihibit (should be a valid Rust identifier)."""
    mat: np.ndarray[Any, Any]
    """Matrix for the Netbrot set."""

    upper_left: complex
    """Upper left corner of the rendering bounding box."""
    lower_right: complex
    """Lower right corner of the rendering bounding box."""

    max_escape_radius: float
    """Maximum desired escape radius. This is meant as a hack around matrices
    where a good estimate is not available.
    """

    def __post_init__(self) -> None:
        assert self.mat.ndim == 2
        assert self.mat.shape[0] == self.mat.shape[1]
        assert self.max_escape_radius > 0.0

    @property
    def identifier(self) -> str:
        return self.name.upper()

    @property
    def size(self) -> int:
        return self.mat.shape[0]

    @property
    def escape_radius_estimate(self) -> float:
        n = self.size
        sigma = np.linalg.svdvals(self.mat)

        return 2.0 * np.sqrt(n) / np.min(sigma) ** 2

    @property
    def escape_radius(self) -> float:
        return min(self.max_escape_radius, self.escape_radius_estimate)

    @property
    def stringified_mat(self) -> str:
        n = self.size
        return "\n".join(
            "{};".format(
                ", ".join(f"c64!({float(self.mat[i, j])!r})" for j in range(n))
            )
            for i in range(n)
        )


def parse_ranges(ranges: str | None) -> list[slice]:
    if ranges is None:
        return []

    slices: set[int] = set()
    for entry in ranges.split(","):
        parts = [part.strip() for part in entry.split(":")]
        nparts = len(parts)

        if nparts == 0:
            continue
        elif nparts == 1:
            try:
                start = int(parts[0])
            except ValueError:
                log.error("Failed to parse range into integer: '%s'", entry)
                continue

            end = start + 1
        elif nparts == 2:
            try:
                start = int(parts[0]) if parts[0] else None
                end = int(parts[1]) if parts[1] else None
            except ValueError:
                log.error("Failed to parse range into integer: '%s'", entry)
                continue
        else:
            raise ValueError(f"Invalid range format: '{entry.strip()}'")

        slices.add(slice(start, end))

    return list(slices)


def make_jinja_env() -> jinja2.Environment:
    env = jinja2.Environment(
        block_start_string="(((",
        block_end_string=")))",
        variable_start_string="((*",
        variable_end_string="*))",
        comment_start_string="((=",
        comment_end_string="=))",
        autoescape=True,
    )

    return env


def main(
    infiles: list[pathlib.Path],
    outfile: pathlib.Path | None = None,
    *,
    slices: list[slice] | None = None,
    max_escape_radius: float = np.inf,
    overwrite: bool = False,
) -> int:
    if not overwrite and outfile is not None and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    if slices is None:
        slices = [slice(None) for _ in infiles]

    # {{{ read matrices

    ret = 0
    exhibits = []

    for filename, fslices in zip(infiles, slices):
        if not filename.exists():
            ret = 1
            log.error("File does not exist: '%s'.", filename)
            continue

        data = np.load(filename, allow_pickle=True)

        matrices = data["matrices"]
        upper_left = data["upper_lefts"]
        lower_right = data["lower_rights"]

        indices = set()
        for s in fslices:
            indices.update(range(*s.indices(matrices.size)))

        if not indices:
            ret = 1
            log.error("No indices in range for '%s'.", filename)
            continue

        suffix = filename.stem.upper().replace("-", "_")
        width = len(str(matrices.size))

        for i in sorted(indices):
            mat = matrices[i]
            ex = Exhibit(
                name=f"EXHIBIT_{i:0{width}}_{suffix}".upper(),
                mat=mat,
                upper_left=complex(upper_left[i]),
                lower_right=complex(lower_right[i]),
                max_escape_radius=max_escape_radius,
            )

            exhibits.append(ex)
            log.info(
                "Loaded exhibit %3d '%s': shape %s (cond %.3e) "
                "escape radius %g (estimate %.3e)",
                i,
                ex.name,
                ex.mat.shape,
                np.linalg.cond(mat),
                ex.escape_radius,
                ex.escape_radius_estimate,
            )

    # }}}

    env = make_jinja_env()
    result = env.from_string(TEMPLATE).render(exhibits=exhibits)

    if outfile:
        with open(outfile, "w", encoding="utf-8") as outf:
            outf.write(result)
    else:
        print(result)

    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filenames",
        nargs="+",
        help="List of (ranged) filenames with the format 'filename.npz[@1,4:10]'",
    )

    parser.add_argument("-o", "--outfile", type=pathlib.Path, default=None)
    parser.add_argument(
        "--max-escape-radius",
        type=float,
        default=np.inf,
        help="Desired maximum escape radius for the infile data",
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

    infiles = []
    ranges = []

    for value in args.filenames:
        parts = value.rsplit("@", maxsplit=1)
        if len(parts) == 1:
            infile, rs = parts[0], ":"
        else:
            infile, rs = parts

        infiles.append(pathlib.Path(infile))
        ranges.append(rs)

    raise SystemExit(
        main(
            infiles,
            args.outfile,
            slices=[parse_ranges(rs) for rs in ranges],
            max_escape_radius=args.max_escape_radius,
            overwrite=args.overwrite,
        )
    )
