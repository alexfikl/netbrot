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
pub struct EscapeResult {
    /// Iteration at which the point escaped or None otherwise.
    pub iteration: Option<usize>,
    /// Last point of the iterate (will be very large if the point escaped).
    pub z: Vector,
}

/// Period of a point, if it does not escape.
type PeriodResult = Option<usize>;

// }}}

// {{{ escape

/// Compute the escape time for the quadratic Netbrot map
///
/// $$
///     f(z) = (A z) * (A z) + c,
/// $$
///
/// where $A$ is a $d \times d$ matrix, $z$ is a $d$ dimensional vector and
/// $c$ is a complex constant.
pub fn netbrot_orbit(brot: &Netbrot) -> EscapeResult {
    let mut z = brot.z0.clone();
    let mat = &brot.mat;
    let c = brot.c;
    let maxit = brot.maxit;
    let escape_radius_squared = brot.escape_radius_squared;

    let mut matz = brot.z0.clone();
    mat.mul_to(&z, &mut matz);

    for i in 0..maxit {
        if z.norm_squared() > escape_radius_squared {
            return EscapeResult {
                iteration: Some(i),
                z,
            };
        }

        z = matz.component_mul(&matz).add_scalar(c);
        mat.mul_to(&z, &mut matz);
    }

    EscapeResult {
        iteration: None,
        z: z.clone(),
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
        EscapeResult { iteration: None, z } => {
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
        EscapeResult {
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
}

// }}}
