# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from typing import Any, Iterator

import numpy as np
import numpy.linalg as la
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

Array = np.ndarray[Any, np.dtype[Any]]
Matrix = Array

# {{{ matplotlib


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


# {{{ netbrot maps


def netbrot(z: Array, mat: Matrix, c: complex, n: int) -> Array:
    for _ in range(n):
        z = (mat @ z) ** 2 + c

    return z


def netbrot_prime(z0: Array, mat: Matrix, c: complex, n: int) -> Array:
    z = z0.reshape(mat.shape[0], -1)
    jac = np.einsum("in,ij->ijn", mat @ z, 2.0 * mat)

    for _ in range(1, n):
        z = (mat @ z) ** 2 + c
        jac = np.einsum(
            "ijn,jkn->ikn",
            np.einsum("in,ij->ijn", mat @ z, 2.0 * mat),
            jac,
        )

    return jac.squeeze()


def netbrot_fp(z: Array, mat: Matrix, c: complex, n: int) -> Array:
    return netbrot(z, mat, c, n) - z


def netbrot_prime_fp(z0: Array, mat: Matrix, c: complex, n: int) -> Array:
    return netbrot_prime(z0, mat, c, n) - np.eye(z0.size)


def netbrot_lsq(z: Array, mat: Matrix, c: complex, n: int) -> Array:
    fz = netbrot_fp(z, mat, c, n)
    return 0.5 * np.sum(fz * fz.conj(), axis=0).real


def netbrot_prime_lsq(z0: Array, mat: Matrix, c: complex, n: int) -> Array:
    fz = netbrot_fp(z0, mat, c, n).reshape(z0.size, -1)
    jacz = netbrot_prime_fp(z0, mat, c, n).reshape(z0.size, z0.size, -1)

    return np.einsum("in,ijn->in", fz, jacz).squeeze()


# }}}


# {{{ main


def divisors(n: int) -> Iterator[int]:
    if n <= 1:
        return

    sqrt_n = int(np.sqrt(n))

    for i in range(1, sqrt_n + 1):
        if n % i == 0:
            yield i


def random_sphere(rng: np.random.Generator, ndim: int) -> Array:
    zs = rng.uniform(0.0, 1.0, size=(2 * ndim,))
    factor = rng.uniform(0.0, 1.0)
    zs = factor ** (1.0 / ndim) * zs / la.norm(zs)

    return zs[:ndim] + 1j * zs[ndim:]


def is_unique_fixed_point(
    fixedpoints: list[Array],
    z: Array,
    mat: Array,
    c: complex,
    nperiod: int,
    *,
    eps: float = 1.0e-15,
) -> bool:
    # 1. Skip point if it is not a fixed point
    is_fp = la.norm(netbrot_fp(z, mat, c, nperiod)) < eps
    if not is_fp:
        return False

    # 2. Skip point if it is already in the list
    isin = any(la.norm(z - zj) < eps for zj in fixedpoints)
    if isin:
        return False

    # 3. Skip point if it is also of a lower period
    is_lower_period = any(
        la.norm(netbrot_fp(z, mat, c, j)) < eps for j in range(1, nperiod)
    )
    if is_lower_period:  # noqa: SIM103
        return False

    return True


def main(
    filename: pathlib.Path,
    c: complex,
    *,
    nperiod: int = 1,
    npoints: int = 2048,
    check_gradients: bool = False,
) -> int:
    rng = np.random.default_rng(seed=42)

    # {{{ read in matrix

    import json

    if not filename.exists():
        log.error("File does not exist: '%s'.", filename)
        return 1

    with open(filename, encoding="utf-8") as inf:
        data = json.load(inf)

    escape_radius = data["escape_radius"] + 1
    elements, *shape = data["mat"]
    assert shape[0] == shape[1]

    mat = np.array([complex(*e) for e in elements]).reshape(*shape).T
    ndim = shape[1]

    nbezout = (2**ndim) ** nperiod
    log.info("Bezout number: %d", nbezout)

    nunique = nbezout - sum(((2**ndim) ** j for j in divisors(nperiod)), 0)
    log.info("Unique solutions: %d", nunique)

    # }}}

    # {{{ find fixed points

    import scipy.optimize as so

    eps = 1.0e-5
    maxit = 10_000
    ntries = 0

    fixedpoints = []
    while ntries < npoints:
        zs = escape_radius * random_sphere(rng, ndim)
        result = so.root(
            netbrot_fp,
            zs,
            args=(mat, c, nperiod),
            # NOTE: working methods:
            #   broyden1, broyden2
            method="broyden1",
            jac=netbrot_prime_fp,
            options={
                "maxfev": maxit,
                "fatol": eps / 10,
                "xatol": eps / 10,
                "maxiter": maxit,
            },
        )
        ntries += 1

        if not result.success:
            continue

        zstar = result.x
        log.info(
            "[%04d/%02d] Message: %s (z0 %s)",
            ntries,
            len(fixedpoints),
            result.message,
            zs,
        )
        log.info("                zstar %s", zstar)
        log.info(
            "                norm %.8e error %.8e jac %.8e",
            la.norm(result.x),
            la.norm(netbrot_fp(zstar, mat, c, nperiod)),
            0.0,  # la.norm(result.jac),
        )

        if is_unique_fixed_point(fixedpoints, zstar, mat, c, nperiod, eps=eps):
            fixedpoints.append(np.real_if_close(zstar, tol=eps))

        if len(fixedpoints) == nunique:
            break

    log.info("Found %d out of %d roots", len(fixedpoints), nunique)
    for i, z_i in enumerate(fixedpoints):
        log.info(
            "z^*_{%d}: (error %.8e) %s",
            i,
            la.norm(netbrot_fp(z_i, mat, c, nperiod)),
            z_i,
        )

    # }}}

    import matplotlib.pyplot as mp

    set_recommended_matplotlib()
    fig = mp.figure()
    ax = fig.gca()

    fp = np.array(fixedpoints).T
    error = la.norm(fp.reshape(ndim, -1, 1) - fp.reshape(ndim, 1, -1), axis=0)
    assert error.shape == (fp.shape[1], fp.shape[1])

    im = ax.imshow(np.log10(error + eps))
    ax.set_title(r"$\log_{10} \|z_i^* - z_j^*\|_2$")
    fig.colorbar(im, ax=ax)

    outfile = filename.parent / f"{filename.stem}-error"
    fig.savefig(outfile)
    mp.close(fig)

    log.info("Saved output for 'c=%s' to '%s'.", c, outfile)


# }}}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=pathlib.Path)
    parser.add_argument(
        "-c",
        nargs=2,
        type=float,
        default=(0, 0),
        help="Real and imaginary parts of c",
    )
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

    raise SystemExit(main(args.filename, complex(args.c[0], args.c[1]), nperiod=args.n))
