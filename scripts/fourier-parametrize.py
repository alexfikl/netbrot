# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager, suppress
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
This script extracts a Fourier description of the boundary of a given fractal
image generate by the main netbrot program. As expected, this is not going to
catch much in the way of the fractal nature of the object, but it does give a
reasonable approximation.

We use the standard edge detection to get the countour and apply the
Ramer-Douglas-Peucker algorithm to approximate the contour with a smaller number
of line segments. We then apply a standard FFT to this approximation to obtain
the Fourier modes.

Example:

    > {SCRIPT_PATH.name} --bbox -1.0 1.0 -1.0 1.0 exhibit-render.png
"""

Array = np.ndarray[Any, np.dtype[Any]]
Scalar = np.floating[Any] | np.complexfloating[Any]


# {{{ plotting settings


def set_recommended_matplotlib() -> None:
    import matplotlib.pyplot as mp

    with suppress(ImportError):
        import scienceplots  # noqa: F401

        mp.style.use(["science", "ieee"])

    mp.style.use(SCRIPT_PATH.parent / "default.mplstyle")


@contextmanager
def axis(filename: pathlib.Path) -> Iterator[Any]:
    import matplotlib.pyplot as mp

    fig = mp.figure(num=1)
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


def test_curve_circle() -> bool:
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

    return True


# }}}


# {{{ parametrize


def lerp(x: float, *, xfrom: tuple[float, float], xto: tuple[float, float]) -> float:
    a, b = xfrom
    t, s = xto

    return t + (x - a) / (b - a) * (s - t)


def resample(modes: Array, n: int) -> Array:
    if n == 1:
        result = modes[0:1].copy()
        return result

    if modes.size == 1:
        result = np.zeros(n, dtype=modes.dtype)
        result[0] = modes[0]
        return result

    m = modes.size // 2
    fac = n / (2 * m)

    if n < m:
        result = np.fft.fftshift(modes)
        result = result[m - n // 2 : m + n // 2]
        result = fac * np.fft.ifftshift(result)
    else:
        result = np.zeros(n, dtype=modes.dtype)
        result[:m] = fac * modes[:m]
        result[-m:] = fac * modes[-m:]

    return result


def parametrize_fourier(
    filenames: list[pathlib.Path],
    *,
    bbox: tuple[float, float, float, float],
    eps: float = 5.0e-4,
    overwrite: bool = False,
    debug: bool = False,
) -> Array:
    import cv2

    xmin, xmax, ymin, ymax = bbox
    results = []

    for filename in filenames:
        if not filename.exists():
            log.error("File does not exist: '%s'.", filename)
            continue

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
        log.debug(
            "> Contour area %.5e perimeter %.5e points %d",
            cv2.contourArea(c),
            perimeter,
            len(c),
        )

        if debug:
            # draw contours
            output = img.copy()
            cv2.drawContours(output, [c], -1, (0, 0, 255), 3)
            cv2.imwrite(filename.with_stem(f"{filename.stem}-fourier-orig"), output)

        # approximate the contour
        approx = cv2.approxPolyDP(c, perimeter * eps, closed=True)
        log.debug("> Approx tolerance %.5e points %d", perimeter * eps, len(approx))

        if debug:
            # draw approximated contour
            output = img.copy()
            cv2.drawContours(output, [approx], -1, (0, 0, 255), 3)
            cv2.imwrite(filename.with_stem(f"{filename.stem}-fourier-approx"), output)

        # get interface points as complex variables in the given bbox
        x = lerp(approx[:, 0, 0], xfrom=(0, img.shape[0]), xto=(xmin, xmax))
        y = lerp(approx[:, 0, 1], xfrom=(0, img.shape[1]), xto=(ymin, ymax))
        z = x + 1j * y

        # NOTE: roll the coefficients until the first one is on the y=0 line
        while np.sign(y[0]) * z[0].imag > 0:
            z = np.roll(z, 1)
        z = np.roll(z, -1)

        # get the Fourier modes
        zhat = np.fft.fft(z)

        # draw Fourier modes
        if debug:
            k = np.fft.fftshift(np.fft.fftfreq(zhat.size, d=1.0 / zhat.size))

            with axis(filename.with_stem(f"{filename.stem}-fourier-modes")) as ax:
                ax.semilogy(k, np.fft.fftshift(np.abs(zhat.real)), "o-", label="Real")
                ax.semilogy(k, np.fft.fftshift(np.abs(zhat.imag)), "o-", label="Imag")
                ax.semilogy(k, 1.0 / np.abs(k) ** 1.01, "k-")
                ax.legend()

        # draw Fourier contour
        if debug:
            for n in [2, 4, 6, 8, 10, 12, 16, 24, 32, 40, 48, 56, 64]:
                with axis(
                    filename.with_stem(f"{filename.stem}-fourier-contour-{n:02d}")
                ) as ax:
                    zfine = resample(resample(zhat, n), 4 * zhat.size)
                    zfine = np.fft.ifft(zfine)

                    ax.plot(z.real, z.imag, "o-", ms=2)
                    ax.plot(zfine.real, zfine.imag, "-")
                    ax.plot(z[0].real, z[0].imag, "o")
                    ax.plot(z[-1].real, z[-1].imag, "o")

                    ax.set_xlabel("$x$")
                    ax.set_ylabel("$y$")
                    ax.set_xlim([xmin, xmax])
                    ax.set_ylim([ymin, ymax])
                    ax.set_title(rf"\# modes = {n} / {zhat.size}")

        results.append(zhat)

    result = np.empty(len(results), dtype=object)
    for i, value in enumerate(results):
        result[i] = value

    return result


def save_geometry(
    filenames: list[pathlib.Path],
    outfile: pathlib.Path | None = None,
    *,
    bbox: tuple[float, float, float, float],
    nmodes: int | None,
    eps: float = 5.0e-4,
    overwrite: bool = False,
    debug: bool = False,
) -> None:
    if outfile is None:
        outfile = pathlib.Path(f"{SCRIPT_PATH.stem}-results.npz")

    modes = parametrize_fourier(
        filenames,
        bbox=bbox,
        eps=eps,
        overwrite=overwrite,
        debug=debug,
    )

    centroids = np.empty(modes.size, dtype=np.complex128)
    areas = np.empty(modes.size)
    perimeters = np.empty(modes.size)
    distances = np.empty(modes.size, dtype=object)
    normals = np.empty(modes.size, dtype=object)
    curvatures = np.empty(modes.size, dtype=object)

    for i, mode in enumerate(modes):
        if nmodes is not None:
            mode = resample(mode, nmodes)  # noqa: PLW2901
        log.info("Computing geometry for exhibit %d with %d modes", i, mode.size)

        curve = curve_geometry(mode)

        if debug:
            filename = outfile.with_stem(f"{outfile.stem}-{i:02d}-normal")
            with axis(filename.with_suffix("")) as ax:
                z = curve.z
                ax.plot(z.real, z.imag)

                cs = curve_geometry(resample(mode, 96))
                zs = cs.z
                ns = cs.normal
                ax.plot(zs.real, zs.imag, "o-", lw=1)
                ax.quiver(zs.real, zs.imag, ns.real, ns.imag)
                if bbox:
                    ax.set_xlim([bbox[0], bbox[1]])
                    ax.set_ylim([bbox[2], bbox[3]])

        centroids[i] = curve.centroid
        areas[i] = curve.area
        perimeters[i] = curve.perimeter

        distances[i] = curve.centroid_distance
        normals[i] = curve.normal
        curvatures[i] = curve.kappa

    np.savez(
        outfile,
        bbox=bbox,
        modes=modes,
        centroids=centroids,
        areas=areas,
        perimeters=perimeters,
        distances=distances,
        normals=normals,
        curvatures=curvatures,
    )
    log.info("Saving geometry information: '%s'.", outfile)


# }}}


# {{{ export


def main(
    filenames: list[pathlib.Path],
    outfile: pathlib.Path | None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    nmodes: int | None = None,
    eps: float = 5.0e-4,
    overwrite: bool = False,
    debug: bool = False,
) -> int:
    try:
        import cv2  # noqa: F401
    except ImportError:
        log.error("'cv2' package not found.")
        return 1

    try:
        import matplotlib.pyplot as mp  # noqa: F401
    except ImportError:
        log.error("'matplotlib' package not found.")
        return 1

    if bbox is None:
        bbox = (-1.0, 1.0, -1.0, 1.0)

    set_recommended_matplotlib()

    # {{{ gather geometry information

    if not (len(filenames) == 1 and filenames[0].suffix == ".npz"):
        if not overwrite and outfile.exists():
            log.error("Output file exists (use --overwrite): '%s'.", outfile)
            return 1

        save_geometry(
            filenames,
            outfile,
            bbox=bbox,
            nmodes=nmodes,
            eps=eps,
            overwrite=overwrite,
            debug=debug,
        )
        filename = outfile
    else:
        (filename,) = filenames

    data = np.load(filename, allow_pickle=True)
    centroids = data["centroids"]
    distances = data["distances"]
    curvatures = data["curvatures"]
    perimeters = data["perimeters"]
    areas = data["areas"]

    # }}}

    # {{{ plot

    outfile = outfile.with_suffix("")

    with axis(outfile.with_stem(f"{outfile.stem}-centroid")) as ax:
        ax.plot(centroids.real, centroids.imag, "o")
        ax.axvline(0.0, color="k", ls="--")
        ax.axhline(0.0, color="k", ls="--")

        offset = 0.0001 * la.norm(centroids, ord=np.inf)
        for i, c in enumerate(centroids):
            ax.text(c.real + offset, c.imag + offset, f"{i}", fontsize=10)

        offset = 0.01 * la.norm(centroids.real, ord=np.inf)
        xmin = min(0.0, np.min(centroids.real)) - offset
        xmax = max(np.max(centroids.real), 0.0) + offset
        ax.set_xlim([xmin, xmax])

        offset = 0.01 * la.norm(centroids.imag, ord=np.inf)
        ymin = min(0.0, np.min(centroids.imag)) - offset
        ymax = max(np.max(centroids.imag), 0.0) + offset
        ax.set_ylim([ymin, ymax])

        ax.set_xlabel("$c_x$")
        ax.set_ylabel("$c_y$")

    with axis(outfile.with_stem(f"{outfile.stem}-centroid-histogram")) as ax:
        ax.hist2d(centroids.real, centroids.imag, bins=(8, 8), density=False)
        ax.set_xlabel("$c_x$")
        ax.set_ylabel("$c_y$")
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

    with axis(outfile.with_stem(f"{outfile.stem}-distance")) as ax:
        n = np.arange(distances.size)
        d_mean = np.array([np.mean(d) for d in distances])
        d_std = np.array([np.std(d) for d in distances])

        ax.fill_between(n, d_mean + d_std, d_mean - d_std, alpha=0.2)
        ax.plot(n, d_mean, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-curvature")) as ax:
        n = np.arange(curvatures.size)
        kappa = np.array([np.median(k) for k in curvatures])

        ax.plot(n, kappa, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-curvature-histogram")) as ax:
        ax.hist(kappa, bins=16, density=False, rwidth=0.8)

    with axis(outfile.with_stem(f"{outfile.stem}-perimeter")) as ax:
        ax.plot(perimeters, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-perimeter-histogram")) as ax:
        ax.hist(perimeters, bins=16, density=False, rwidth=0.8)

    with axis(outfile.with_stem(f"{outfile.stem}-area")) as ax:
        ax.plot(areas, "o-")

    with axis(outfile.with_stem(f"{outfile.stem}-area-histogram")) as ax:
        ax.hist(areas, bins=16, density=False, rwidth=0.8)

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
    parser.add_argument("filenames", nargs="*", type=pathlib.Path)
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
        default=(-1, 1, -1, 1),
        help="The bounding box in physical coordinates for the images",
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

    if args.debug:
        log.setLevel(logging.DEBUG)

    assert test_curve_circle()
    raise SystemExit(
        main(
            args.filenames,
            args.outfile,
            bbox=args.bbox,
            nmodes=args.modes,
            overwrite=args.overwrite,
            debug=args.debug,
        )
    )
