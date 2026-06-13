// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::{Complex64, c64};

use nalgebra::{DMatrix, DVector};

use crate::render::{MAX_PERIODS, PERIOD_WINDOW};

// {{{ types

pub type Matrix = DMatrix<Complex64>;
pub type Vector = DVector<Complex64>;

#[derive(Clone, Debug)]
pub struct Netbrot {
    /// Matrix used in the iteration
    pub mat: Matrix,
    /// Starting point for the iteration.
    pub z0: Vector,
    /// Constant offset for the iteration.
    pub c: Complex64,

    /// Maximum number of iterations before the point is considered in the set.
    pub maxit: usize,
    /// Estimated escape radius (squared).
    pub escape_radius_squared: f64,
}

impl Netbrot {
    pub fn new(mat: &Matrix, maxit: usize, escape_radius: f64) -> Self {
        Netbrot {
            mat: mat.clone(),
            z0: Vector::zeros(mat.nrows()),
            c: c64(0.0, 0.0),
            maxit,
            escape_radius_squared: escape_radius * escape_radius,
        }
    }

    pub fn evaluate(&self, z: &Vector) -> Vector {
        let mut out = z.clone_owned();
        self.evaluate_to(z, &mut out);

        out
    }

    pub fn evaluate_to(&self, z: &Vector, out: &mut Vector) {
        self.mat.mul_to(z, out);
        for e in out.iter_mut() {
            *e = *e * *e + self.c;
        }
    }

    pub fn jacobian(&self, z: &Vector) -> Matrix {
        // https://github.com/dimforge/nalgebra/issues/1338
        let matz = (&self.mat * z) * c64(2.0, 0.0);

        Matrix::from_diagonal(&matz) * &self.mat
    }

    #[allow(dead_code)]
    pub fn jacobian_to(&self, z: &Vector, out: &mut Matrix) {
        let matz = (&self.mat * z) * c64(2.0, 0.0);
        Matrix::from_diagonal(&matz).mul_to(&self.mat, out);
    }
}

#[derive(Clone, Debug)]
pub struct PeriodEscapeResult {
    /// Iteration at which the point escaped or None otherwise.
    pub iteration: Option<usize>,
    /// Last point of the iterate (will be very large if the point escaped).
    pub z: Vector,
}

/// Escape-time result optimized for rendering (avoids retaining the full state vector).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EscapeResult {
    /// Iteration at which the point escaped, or `None` if it stayed bounded.
    pub iteration: Option<usize>,
    /// Euclidean norm of *z* when escaped (only meaningful if `iteration` is `Some`).
    pub z_norm: f64,
}

/// Period of a point, if it does not escape.
type PeriodResult = Option<usize>;

// }}}

// {{{ escape

/// Specialized 1D orbit (scalar complex state).
#[inline]
pub fn netbrot_orbit_escape_1d(
    a: Complex64,
    z: Complex64,
    c: Complex64,
    maxit: usize,
    escape_radius_squared: f64,
) -> EscapeResult {
    let mut z = z;
    let mut w = a * z;

    for i in 0..maxit {
        let norm_sq = z.norm_sqr();
        if norm_sq > escape_radius_squared {
            return EscapeResult {
                iteration: Some(i),
                z_norm: norm_sq.sqrt(),
            };
        }

        z = w * w + c;
        w = a * z;
    }

    EscapeResult {
        iteration: None,
        z_norm: 0.0,
    }
}

/// Specialized 2D orbit with stack-allocated state (no heap allocations in the loop).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn netbrot_orbit_escape_2d(
    a00: Complex64,
    a01: Complex64,
    a10: Complex64,
    a11: Complex64,
    z0: Complex64,
    z1: Complex64,
    c: Complex64,
    maxit: usize,
    escape_radius_squared: f64,
) -> EscapeResult {
    let mut z0 = z0;
    let mut z1 = z1;
    let mut w0 = a00 * z0 + a01 * z1;
    let mut w1 = a10 * z0 + a11 * z1;

    for i in 0..maxit {
        let norm_sq = z0.norm_sqr() + z1.norm_sqr();
        if norm_sq > escape_radius_squared {
            return EscapeResult {
                iteration: Some(i),
                z_norm: norm_sq.sqrt(),
            };
        }

        z0 = w0 * w0 + c;
        z1 = w1 * w1 + c;
        w0 = a00 * z0 + a01 * z1;
        w1 = a10 * z0 + a11 * z1;
    }

    EscapeResult {
        iteration: None,
        z_norm: 0.0,
    }
}

/// General N-dimensional orbit with reusable buffers (no per-iteration allocations).
pub fn netbrot_orbit_escape_ndim(
    mat: &Matrix,
    z: &mut Vector,
    c: Complex64,
    maxit: usize,
    escape_radius_squared: f64,
    matz: &mut Vector,
) -> EscapeResult {
    mat.mul_to(z, matz);

    for i in 0..maxit {
        let norm_sq: f64 = z.iter().map(|zi| zi.norm_sqr()).sum();
        if norm_sq > escape_radius_squared {
            return EscapeResult {
                iteration: Some(i),
                z_norm: norm_sq.sqrt(),
            };
        }

        for (zi, wi) in z.iter_mut().zip(matz.iter()) {
            *zi = wi * wi + c;
        }
        mat.mul_to(z, matz);
    }

    EscapeResult {
        iteration: None,
        z_norm: 0.0,
    }
}

/// Compute the escape time for the quadratic Netbrot map
///
/// $$
///     f(z) = (A z) * (A z) + c,
/// $$
///
/// where $A$ is a $d \times d$ matrix, $z$ is a $d$ dimensional vector and
/// $c$ is a complex constant.
pub fn netbrot_orbit(brot: &Netbrot) -> PeriodEscapeResult {
    let mut z = brot.z0.clone();
    let mut matz = Vector::zeros(z.len());
    let escape = netbrot_orbit_escape_ndim(
        &brot.mat,
        &mut z,
        brot.c,
        brot.maxit,
        brot.escape_radius_squared,
        &mut matz,
    );

    PeriodEscapeResult {
        iteration: escape.iteration,
        z,
    }
}

// }}}

// {{{ period

/// Compute the period of a point from the set.
///
/// The period is computed by looking at a long time iteration that does not
/// escape and checking the tolerance.
pub fn netbrot_orbit_period(brot: &Netbrot) -> PeriodResult {
    match netbrot_orbit(brot) {
        PeriodEscapeResult { iteration: None, z } => {
            // When the limit was reached but the point did not escape, we look
            // for a period in a very naive way.
            let mat = &brot.mat;
            let c = brot.c;
            let mut matz = z.clone();
            let mut z_period: Vec<Vector> = Vec::with_capacity(PERIOD_WINDOW);

            // Evaluate some more points
            z_period.push(z.clone());
            mat.mul_to(&z, &mut matz);

            #[allow(clippy::needless_range_loop)]
            for i in 1..PERIOD_WINDOW {
                z_period.push(matz.component_mul(&matz).add_scalar(c));
                mat.mul_to(&z_period[i], &mut matz);
            }

            // Check newly evaluated points for periodicity
            for i in 2..MAX_PERIODS {
                let mut z_period_norm: f64 = 0.0;
                for j in 0..i - 1 {
                    let zj = &z_period[j];
                    let zi = &z_period[i + j - 1];
                    z_period_norm += (zj - zi).norm_squared();
                }

                if z_period_norm.sqrt() < 1.0e-3 {
                    return Some(i - 1);
                }
            }

            Some(MAX_PERIODS - 1)
        }
        PeriodEscapeResult {
            iteration: Some(_),
            z: _,
        } => None,
    }
}

// }}}

// {{{ tests

#[cfg(test)]
mod tests {
    use super::*;

    use nalgebra::dmatrix;
    use num::complex::c64;

    use crate::fixedpoints::generate_random_points_in_ball;

    #[test]
    fn test_zero_escape() {
        let maxit = 512;
        let escape_radius = 5.0;
        let mat = dmatrix![c64(1.0, 0.0), c64(0.8, 0.0); c64(1.0, 0.0), c64(-0.5, 0.0)];

        let brot = Netbrot::new(&mat, maxit, escape_radius);
        let mut z = brot.z0.clone_owned();
        let mut znext = z.clone_owned();

        for _ in 0..brot.maxit {
            brot.evaluate_to(&z, &mut znext);
            z.copy_from(&znext);

            let znext_copy = brot.evaluate(&z);
            assert!((&znext - &znext_copy).norm() < 1.0e-15);
        }

        // c = 0 should not escape for this fractal
        assert!(z.norm_squared() < brot.escape_radius_squared);
    }

    #[test]
    fn test_jacobian_vs_finite_difference() {
        let ndim = 2;
        let maxit = 512;
        let escape_radius = 5.0;

        let mat = dmatrix![c64(1.0, 0.0), c64(0.8, 0.0); c64(1.0, 0.0), c64(-0.5, 0.0)];
        let brot = Netbrot::new(&mat, maxit, escape_radius);

        let mut fz = brot.z0.clone_owned();
        let mut fz_eps = brot.z0.clone_owned();

        let mut jac = mat.clone_owned();
        let mut jac_est = mat.clone_owned();
        let mut rng = rand::rng();

        let basis = Matrix::identity(ndim, ndim);
        let mut err = DVector::<f64>::zeros(7);
        let eps = DVector::<f64>::from_fn(err.len(), |i, _| 10.0_f64.powi(-(i as i32)));

        for _ in 0..32 {
            let z = generate_random_points_in_ball(&mut rng, ndim, escape_radius);

            // evaluate
            brot.evaluate_to(&z, &mut fz);
            brot.jacobian_to(&z, &mut jac);

            let jac_norm = jac.norm();
            let jac_copy = brot.jacobian(&z);
            assert!((&jac - &jac_copy).norm() < 1.0e-15 * jac_norm);

            // FIXME: copy this out into a little function? in newton.rs?
            for n in 0..err.len() {
                for j in 0..ndim {
                    let z_eps = &z + basis.column(j).scale(eps[n]);
                    brot.evaluate_to(&z_eps, &mut fz_eps);

                    for i in 0..ndim {
                        jac_est[(i, j)] = (fz_eps[i] - fz[i]) / eps[n];
                    }
                }

                err[n] = (&jac - &jac_est).norm() / jac_norm;
            }

            let order = DVector::from_iterator(
                err.len() - 1,
                (0..err.len() - 1).map(|i| {
                    (err[i + 1].log2() - err[i].log2()) / (eps[i + 1].log2() - eps[i].log2())
                }),
            );

            println!("Order: {}", order.min());
            assert!(order.min() > 0.9);
        }
    }
    #[test]
    fn test_escape_kernels_match() {
        let maxit = 100;
        let escape_radius = 5.0;
        let escape_r2 = escape_radius * escape_radius;

        // --- Test 1D ---
        let mat1d = dmatrix![c64(0.5, 0.5)];
        let c1d = c64(-0.5, 0.0);
        let z0_1d = c64(0.1, -0.1);
        let mut vec_z_1d = DVector::from_element(1, z0_1d);
        let mut matz_1d = Vector::zeros(1);

        let esc_1d = netbrot_orbit_escape_1d(mat1d[(0, 0)], z0_1d, c1d, maxit, escape_r2);
        let esc_nd_1d =
            netbrot_orbit_escape_ndim(&mat1d, &mut vec_z_1d, c1d, maxit, escape_r2, &mut matz_1d);
        assert_eq!(esc_1d.iteration, esc_nd_1d.iteration);
        assert!((esc_1d.z_norm - esc_nd_1d.z_norm).abs() < 1e-12);

        // --- Test 2D ---
        let mat2d = dmatrix![c64(0.5, 0.0), c64(-0.2, 0.1); c64(0.1, -0.1), c64(0.8, 0.0)];
        let c2d = c64(-0.4, 0.2);
        let z0_2d = [c64(0.2, 0.2), c64(-0.1, 0.0)];
        let mut vec_z_2d = DVector::from_vec(z0_2d.to_vec());
        let mut matz_2d = Vector::zeros(2);

        let esc_2d = netbrot_orbit_escape_2d(
            mat2d[(0, 0)],
            mat2d[(0, 1)],
            mat2d[(1, 0)],
            mat2d[(1, 1)],
            z0_2d[0],
            z0_2d[1],
            c2d,
            maxit,
            escape_r2,
        );
        let esc_nd_2d =
            netbrot_orbit_escape_ndim(&mat2d, &mut vec_z_2d, c2d, maxit, escape_r2, &mut matz_2d);

        assert_eq!(esc_2d.iteration, esc_nd_2d.iteration);
        assert!((esc_2d.z_norm - esc_nd_2d.z_norm).abs() < 1e-12);
    }
}

// }}}
