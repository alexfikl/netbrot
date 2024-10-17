// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::{c64, Complex64};
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::iterate::{netbrot_repeat, netbrot_repeat_prime, Matrix, Netbrot, Vector};
use crate::newton::NewtonRaphson;

/// {{{ polynomial

fn bezout_number(ndim: u32, n: u32) -> u32 {
    2_u32.pow(ndim).pow(n)
}

fn unique_poly_solutions(ndim: u32, n: u32) -> u32 {
    let b = bezout_number(ndim, n);
    let sqrt_n = (n as f64).sqrt() as u32;
    b - (1..sqrt_n + 1)
        .filter(|i| n % i == 0)
        .map(|i| 2_u32.pow(ndim).pow(i))
        .sum::<u32>()
}

/// }}}

/// {{{ find_unique_fixed_points

fn is_unique_fixed_point(
    fixedpoints: &[Vector],
    z: &Vector,
    brot: &Netbrot,
    nperiod: u32,
    eps: f64,
) -> bool {
    if fixedpoints.is_empty() {
        return true;
    }

    let c = brot.c;
    let mat = &brot.mat;
    let divisors: Vec<u32> = (1..nperiod).filter(|i| nperiod % i == 0).collect();

    // Check if fixed point already exists
    let is_in = fixedpoints.iter().any(|z_j| (z - z_j).norm() < eps);
    if is_in {
        return false;
    }

    // Check if the point is a fixed point
    let is_fp = (netbrot_repeat(mat, z, c, nperiod) - z).norm() < eps;
    if !is_fp {
        return false;
    }

    // Check if it's a fixed point of a lower period
    let is_smaller_period = divisors
        .iter()
        .any(|&j| (z - netbrot_repeat(mat, z, c, j)).norm() < eps);
    if is_smaller_period {
        return false;
    }

    true
}

// }}}

// {{{ find_fixed_points_by_newton

fn generate_random_points_on_sphere<R: Rng + ?Sized>(
    rng: &mut R,
    ndim: usize,
    radius: f64,
) -> Vector {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let factor: f64 = rng.gen();
    let mut components: Vec<f64> = (0..2 * ndim).map(|_| normal.sample(rng)).collect();
    let components_norm = components
        .iter()
        .cloned()
        .reduce(|a, b| a + b * b)
        .unwrap_or(0.0)
        .sqrt();

    components = components
        .iter()
        .map(|c| factor * radius / components_norm * c)
        .collect();
    Vector::from_iterator(
        ndim,
        (0..ndim).map(|i| c64(components[i], components[i + ndim])),
    )
}

fn netbrot_repeat_fp(mat: &Matrix, z: &Vector, c: Complex64, n: u32) -> Vector {
    netbrot_repeat(mat, z, c, n) - z
}

fn netbrot_repeat_prime_fp(mat: &Matrix, z: &Vector, c: Complex64, n: u32) -> Matrix {
    netbrot_repeat_prime(mat, z, c, n) - Matrix::identity(mat.nrows(), mat.nrows())
}

pub fn find_fixed_points_by_newton(
    brot: &Netbrot,
    nperiod: u32,
    maxit: u32,
    eps: f64,
) -> Vec<Vector> {
    let mut rng = rand::thread_rng();

    let ndim = brot.mat.nrows();
    let npoints = unique_poly_solutions(ndim as u32, nperiod) as usize;
    let radius = brot.escape_radius_squared.sqrt();

    let mut ntries = 0;
    let mut fixedpoints: Vec<Vector> = Vec::with_capacity(npoints);

    let f = |z: &Vector| netbrot_repeat_fp(&brot.mat, z, brot.c, nperiod);
    let j = |z: &Vector| netbrot_repeat_prime_fp(&brot.mat, z, brot.c, nperiod);
    let solver = NewtonRaphson::new(f, j)
        .with_rtol(eps / 10.0)
        .with_maxit(512);

    while ntries < maxit {
        let z0 = generate_random_points_on_sphere(&mut rng, ndim, radius);
        match solver.solve(&z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                println!(
                    "[{}/{}] Found a root: {} from {}",
                    ntries,
                    fixedpoints.len(),
                    z,
                    z0
                );
                if is_unique_fixed_point(&fixedpoints, &z, brot, nperiod, eps) {
                    fixedpoints.push(z)
                }
            }
            Err(_) => continue,
        }
        ntries += 1;

        if fixedpoints.len() == npoints {
            break;
        }
    }

    fixedpoints
}

// }}}

// {{{ find_fixed_points_by_iteration

// }}}

// {{{ find_fixed_points_by_polynomial

// }}}
