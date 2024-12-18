# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
import numpy.linalg as la
import rich.logging

log = logging.getLogger(pathlib.Path(__file__).stem)
log.setLevel(logging.ERROR)
log.addHandler(rich.logging.RichHandler())

SCRIPT_PATH = pathlib.Path(__file__)
SCRIPT_LONG_HELP = f"""\
Example:

    > {SCRIPT_PATH.name} fourier-modes.npz
"""

Array = np.ndarray[Any, np.dtype[Any]]
Scalar = np.floating[Any] | np.complexfloating[Any]

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


@contextmanager
def axis(filename: pathlib.Path) -> Iterator[Any]:
    import matplotlib.pyplot as mp

    fig = mp.figure()
    ax = fig.gca()

    try:
        yield ax
    finally:
        log.info("Saving figure in '%s'.", filename)
        fig.savefig(filename)
        mp.close(fig)


# }}}


# {{{ curve


def dot(x: Array, y: Array) -> Array:
    return x.real * y.real + x.imag * y.imag


def integrate(c: Curve, f: Array) -> Scalar:
    # NOTE: integrating using a trapezoidal rule on a closed curve
    w = 1.0 / f.size
    return np.sum(f * c.jacobian * w)


@dataclass
class Curve:
    zhat: Array
    """Fourier modes describing the curve."""
    z: Array
    """Curve coordinates in the physical space."""

    jacobian: Array
    """Jacobian of the transformation at each point *z* (used in quadrature)."""
    normal: Array
    """Normal vector at each point *z*."""
    kappa: Array
    """Curvature at each point *z*."""

    @cached_property
    def area(self) -> Scalar:
        return np.abs(integrate(self, 0.5 * dot(self.z, self.normal)))

    @cached_property
    def perimeter(self) -> Scalar:
        return np.abs(integrate(self, np.ones_like(self.jacobian)))

    @cached_property
    def centroid(self) -> Array:
        return integrate(self, 0.5 * dot(self.z, self.z) * self.normal) / self.area

    @cached_property
    def centroid_distance(self) -> Array:
        return np.abs(self.z - self.centroid)


def curve_geometry(zhat: Array) -> Curve:
    z = np.fft.ifft(zhat)
    k = 1.0j * np.fft.fftfreq(zhat.size, d=1.0 / zhat.size / (2.0 * np.pi))

    dx = np.fft.ifft(k * zhat)
    ddx = np.fft.ifft(k**2 * zhat)

    jac = np.abs(dx)
    normal = 1.0j * dx / jac
    kappa = -(normal.real * ddx.real + normal.imag * ddx.imag) / jac**2

    return Curve(zhat=zhat, z=z, jacobian=jac, normal=normal, kappa=kappa)


def test_curve_circle() -> None:
    rng = np.random.default_rng(seed=42)
    R = rng.uniform(1.0, 10.0)

    centroid = 1.0 + 0.5j
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)[::-1]
    z = centroid + R * np.exp(1j * theta)
    zhat = np.fft.fft(z)

    curve = curve_geometry(zhat)

    error = la.norm(z - curve.z)
    assert error < 1.0e-13, error

    jacobian = 2.0 * np.pi * R
    error = la.norm(jacobian - curve.jacobian)
    assert error < 5.0e-12, error

    normal = (z - centroid) / np.abs(z - centroid)
    error = la.norm(normal - curve.normal)
    assert error < 1.0e-13, error

    kappa = 1 / R
    error = la.norm(kappa - curve.kappa)
    assert error < 5.0e-13, error

    area = np.pi * R**2
    error = la.norm(area - curve.area)
    assert error < 2.0e-13, error

    perimeter = 2.0 * np.pi * R
    error = la.norm(perimeter - curve.perimeter)
    assert error < 1.0e-13, error

    error = la.norm(centroid - curve.centroid)
    assert error < 1.0e-13, error

    distance = np.abs(z - centroid)
    error = la.norm(distance - curve.centroid_distance)
    assert error < 1.0e-13, error


# }}}


def truncate(modes: Array, degree: int) -> Array:
    m = modes.size // 2
    assert degree < modes.size

    modes = np.fft.fftshift(modes)
    modes = modes[m - degree // 2 : m + degree // 2]

    return degree / (2 * m) * np.fft.ifftshift(modes)


def categorize_fourier(
    filename: pathlib.Path,
    outfile: pathlib.Path | None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    overwrite: bool = False,
    debug: bool = False,
) -> int:
    test_curve_circle()

    try:
        import matplotlib.pyplot as mp
    except ImportError:
        log.error("'matplotlib' package not found.")
        return 1

    if outfile is None:
        ext = mp.rcParams["savefig.format"]
        outfile = pathlib.Path(f"{SCRIPT_PATH.stem}-results.{ext}")

    if not overwrite and outfile.exists():
        log.error("Output file exists (use --overwrite): '%s'.", outfile)
        return 1

    set_recommended_matplotlib()

    results = np.load(filename, allow_pickle=True)

    centroids = []
    distances = []
    areas = []
    perimeters = []
    curvatures = []

    for i, modes in enumerate(results["modes"]):
        log.info("Loaded exhibit %d with %d modes", i, modes.size)
        curve = curve_geometry(truncate(modes, modes.size // 2))

        if debug:
            with axis(outfile.with_stem(f"{outfile.stem}-{i:02d}-contour")) as ax:
                z = curve.z
                ax.plot(z.real, z.imag)

                cs = curve_geometry(truncate(modes, 92))
                zs = cs.z
                ns = cs.normal
                ax.plot(zs.real, zs.imag, "o-", lw=1)
                ax.quiver(zs.real, zs.imag, ns.real, ns.imag)
                if bbox:
                    ax.set_xlim([bbox[0], bbox[1]])
                    ax.set_ylim([bbox[2], bbox[3]])

        centroids.append(curve.centroid)
        distances.append(curve.centroid_distance)

        areas.append(curve.area)
        perimeters.append(curve.perimeter)
        curvatures.append(curve.kappa)

    with axis(outfile.with_stem(f"{outfile.stem}-centroid")) as ax:
        c = np.array(centroids)
        ax.plot(c.real, c.imag, "o")
        ax.axvline(0.0, color="k", ls="--")

        offset = 0.0001 * la.norm(c, ord=np.inf)
        for i in range(c.size):
            ax.text(c[i].real + offset, c[i].imag + offset, f"{i}", fontsize=10)

        ax.set_xlim([-0.003, 0.003])

    with axis(outfile.with_stem(f"{outfile.stem}-centroid-histogram")) as ax:
        ax.hist2d(c.real, c.imag, bins=(8, 8), density=False)
        ax.set_xlabel("$c_x$")
        ax.set_ylabel("$c_y$")
        ax.set_xlim([-0.003, 0.003])

    with axis(outfile.with_stem(f"{outfile.stem}-distance")) as ax:
        n = np.arange(len(distances))
        d_mean = np.array([np.mean(d) for d in distances])
        d_std = np.array([np.std(d) for d in distances])

        ax.fill_between(n, d_mean + d_std, d_mean - d_std, alpha=0.2)
        ax.plot(n, d_mean, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-curvature")) as ax:
        n = np.arange(len(curvatures))
        kappa_mean = np.array([np.median(k) for k in curvatures])
        # kappa_std = np.array([np.std(k) for k in curvatures])

        # ax.fill_between(n, kappa_mean + kappa_std, kappa_mean - kappa_std, alpha=0.2)
        ax.plot(n, kappa_mean, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-curvature-histogram")) as ax:
        ax.hist(kappa_mean, bins=16, density=False, rwidth=0.8)

    with axis(outfile.with_stem(f"{outfile.stem}-perimeter")) as ax:
        perimeter = np.array(perimeters)
        ax.plot(perimeter, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-perimeter-histogram")) as ax:
        ax.hist(perimeter, bins=16, density=False, rwidth=0.8)

    with axis(outfile.with_stem(f"{outfile.stem}-area")) as ax:
        area = np.array(areas)
        ax.plot(area, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-area-histogram")) as ax:
        ax.hist(area, bins=16, density=False, rwidth=0.8)

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
    parser.add_argument("filename", type=pathlib.Path)
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
        default=None,
        help="The bounding box in physical coordinates for the images",
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
        categorize_fourier(
            args.filename,
            args.outfile,
            bbox=args.bbox,
            overwrite=args.overwrite,
            debug=args.debug,
        )
    )
