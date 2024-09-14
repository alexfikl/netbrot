// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use image::Rgb;
use num::complex::{c64, Complex64};

use nalgebra::{DMatrix, DVector};

use crate::colorschemes::{get_period_color, get_smooth_orbit_color};

pub const MAX_PERIODS: usize = 20;
const PERIOD_WINDOW: usize = 2 * MAX_PERIODS;

// {{{ structs

type Matrix = DMatrix<Complex64>;
type Vector = DVector<Complex64>;

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
            z0: DVector::from_vec(vec![c64(0.0, 0.0); mat.nrows()]),
            c: c64(0.0, 0.0),
            maxit,
            escape_radius_squared: escape_radius * escape_radius,
        }
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

// {{{ helpers

/// Estimate the escape radius for a given matrix $A$.
///
/// $$
///     R = \frac{2 \sqrt{d}}{\sigma_{\text{min}}(A)^2}.
/// $$
#[allow(dead_code)]
pub fn escape_radius_squared(mat: Matrix) -> f64 {
    // NOTE: singular values are sorted descendingly, so we can just take the last
    // one here without worrying about it too much :D
    let n = mat.nrows();
    let sigma_min_sqr = mat.view((0, 0), (n, n)).singular_values()[n - 1].powi(4);
    let fac_sqr = 4.0 * (n as f64);

    fac_sqr / sigma_min_sqr
}

/// Translate pixel coordinates to physical point coordinates.
///
/// *bounds*: width and height of the image.
/// *upper_left*, *lower_left*: bounding box of the domain.
pub fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex64,
    lower_right: Complex64,
) -> Complex64 {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );
    c64(
        // Why subtraction here? pixel.1 increases as we go down,
        upper_left.re + (pixel.0 as f64) * width / (bounds.0 as f64),
        // but the imaginary component increases as we go up.
        upper_left.im - (pixel.1 as f64) * height / (bounds.1 as f64),
    )
}

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

/// Compute the *n* times composition of the Netbrot quadratic map.
///
/// This just computes the composition and does not iterate to an escape.
pub fn netbrot_compose_n(mat: Matrix, z0: Vector, c: Complex64, n: usize) -> Vector {
    let mut z = z0.clone();
    let mut matz = z0.clone();

    for _ in 0..n {
        z = matz.component_mul(&matz).add_scalar(c);
        mat.mul_to(&z, &mut matz);
    }

    z
}

/// Compute the Jacobian of the Netbrot quadratic map.
///
/// The Jacobian is given by
///
/// $$
///     J_f(z) = 2 diag(A z) A
/// $$
///
/// where $diag(x)$ just gives a matrix with *x* on the diagonal.
pub fn netbrot_compose_prime(mat: &Matrix, z: &Vector, jac: &mut Matrix) {
    let mut matz = z.clone();
    mat.mul_to(z, &mut matz);

    let n = mat.nrows();
    for i in 0..n {
        for j in 0..n {
            jac[(i, j)] = 2.0 * mat[(i, j)] * matz[i];
        }
    }
}

// Compute the eigenvalues of the Jacobian of the *n* times composition.
//
// By the chain rule, the Jacobian is given by
//
// $$
//      J_{f^n}(z) = J_f(f^{n - 1}(z)) J_f(f^{n - 2}(z)) \cdots J_f(z)
// $$
//
// We compute the Jacobian of the composition right-to-left by multiplying the
// resulting matrices as we construct the *n* times composition $f^n(z)$.
pub fn netbrot_orbit_lambda_n(mat: Matrix, z0: Vector, c: Complex64, n: usize) -> Vector {
    let mut z = z0.clone();
    let mut matz = z0.clone();

    let mut jac = mat.clone();
    let mut jac_n = mat.clone();
    let mut tmp = mat.clone();

    // Compute J_f(z)
    netbrot_compose_prime(&mat, &z, &mut jac);

    for _ in 1..n {
        // Compute f^n(z)
        z = matz.component_mul(&matz).add_scalar(c);
        mat.mul_to(&z, &mut matz);

        // Compute J_f(f^n(z))
        netbrot_compose_prime(&mat, &z, &mut jac_n);

        // Left multiply into J_{f^n}
        jac_n.mul_to(&jac, &mut tmp);
        jac.copy_from(&tmp);
    }

    jac.eigenvalues().unwrap()
}

/// Returns true if the orbit has any eigenvalues in the unit circle.
///
/// We basically compute the eigenvalues of J_{f^n}(x) and check if the smallest
/// one is smaller than 1, i.e. at least one one eigenvalue is in the unit circle.
pub fn netbrot_orbit_stable(mat: Matrix, z0: Vector, c: Complex64, n: usize) -> bool {
    let eigs = netbrot_orbit_lambda_n(mat, z0, c, n);
    eigs.camin() < 1.0
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

// {{{ rendering

// {{{ render orbits

pub fn render_orbit(
    pixels: &mut [u8],
    brot: &Netbrot,
    bounds: (usize, usize),
    upper_left: Complex64,
    lower_right: Complex64,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);
    let maxit = brot.maxit;
    let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            local_brot.c = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit(&local_brot) {
                EscapeResult {
                    iteration: None,
                    z: _,
                } => Rgb([0, 0, 0]),
                EscapeResult {
                    iteration: Some(n),
                    z,
                } => get_smooth_orbit_color(n, z.norm(), maxit),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

// }}}

// {{{ render periods

pub fn render_period(
    pixels: &mut [u8],
    brot: &Netbrot,
    bounds: (usize, usize),
    upper_left: Complex64,
    lower_right: Complex64,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);
    let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            local_brot.c = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit_period(&local_brot) {
                None => Rgb([255, 255, 255]),
                Some(period) => get_period_color(period, MAX_PERIODS, 3),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

// }}}

// {{{ render periods

pub fn render_fixed_points(
    pixels: &mut [u8],
    brot: &Netbrot,
    bounds: (usize, usize),
    upper_left: Complex64,
    lower_right: Complex64,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);
    let maxit = brot.maxit;
    let nrows = brot.mat.nrows();
    let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let z0 = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            local_brot.z0 = DVector::from_vec(vec![z0; nrows]);
            let color = match netbrot_orbit(&local_brot) {
                EscapeResult {
                    iteration: None,
                    z: _,
                } => Rgb([0, 0, 0]),
                EscapeResult {
                    iteration: Some(n),
                    z,
                } => get_smooth_orbit_color(n, z.norm(), maxit),
            };
            let index = row * bounds.0 + 3 * column;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

// }}}

// }}}
