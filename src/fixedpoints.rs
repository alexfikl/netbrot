// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::{c64, Complex64};
use rand::Rng;

use eqsolver::multivariable::GaussNewton;

use crate::netbrot::{netbrot_repeat, netbrot_repeat_prime, Matrix, Netbrot, Vector};

/// {{{ find_unique_fixed_points

fn find_unique_fixed_points(
    brot: &Netbrot,
    fixedpoints: &[Vector],
    nperiod: u32,
    eps: f64,
) -> Vec<usize> {
    let c = brot.c;
    let mat = &brot.mat;

    let divisors: Vec<u32> = (1..nperiod).filter(|i| nperiod % i == 0).collect();
    let mut result: Vec<usize> = Vec::with_capacity(4.0f32.powi(nperiod as i32) as usize);
    let npoints = fixedpoints.len();

    for i in 0..npoints {
        let z_i = &fixedpoints[i];

        // Check if fixed point already exists
        let is_in = result.iter().any(|&j| (z_i - &fixedpoints[j]).norm() < eps);
        if is_in {
            continue;
        }

        // Check if the point is a fixed point
        let is_fp = (netbrot_repeat(mat, z_i, c, nperiod) - z_i).norm() < eps;
        if !is_fp {
            continue;
        }

        // Check if it's a fixed point of a lower period
        let is_smaller_period = divisors
            .iter()
            .any(|&j| (z_i - netbrot_repeat(mat, z_i, c, j)).norm() < eps);
        if is_smaller_period {
            continue;
        }

        result.push(i);
    }

    result
}

// }}}

// {{{ find_fixed_points_by_newton

fn generate_random_points_on_sphere<R: Rng + ?Sized>(
    rng: &mut R,
    ndim: usize,
    radius: f64,
) -> Vector {
    let factor: f64 = rng.gen();
    let mut components: Vec<f64> = (0..2 * ndim).map(|_| rng.gen()).collect();
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
    npoints: u32,
    nperiod: u32,
    eps: f64,
) -> Vec<Vector> {
    let mut rng = rand::thread_rng();

    let ndim = brot.mat.nrows();
    let radius = brot.escape_radius_squared.sqrt();

    let mut fixedpoints: Vec<Vector> = Vec::with_capacity(npoints as usize);

    let f = |z: Vector| netbrot_repeat_fp(&brot.mat, &z, brot.c, nperiod);
    let j = |z: Vector| netbrot_repeat_prime_fp(&brot.mat, &z, brot.c, nperiod);
    let solver = GaussNewton::new(f, j);

    for _ in 0..npoints {
        let z0 = generate_random_points_on_sphere(&mut rng, ndim, radius);
        let result = solver.solve(z0).unwrap();

        fixedpoints.push(z0);
    }

    let indices = find_unique_fixed_points(brot, &fixedpoints, nperiod, eps);
    indices.iter().map(|&i| fixedpoints[i].clone()).collect()
}

// }}}

// {{{ find_fixed_points_by_iteration

// }}}

// {{{ find_fixed_points_by_polynomial

// }}}
