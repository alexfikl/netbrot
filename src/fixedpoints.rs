// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::c64;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::iterate::{netbrot_repeat, netbrot_repeat_prime, Matrix, Netbrot, Vector};
use crate::newton::{NewtonRaphson, NewtonRaphsonResult};

/// {{{ polynomial solutions

fn bezout_number(ndim: u32, n: u32) -> u32 {
    2_u32.pow(ndim).pow(n)
}

fn unique_poly_solutions(ndim: u32, n: u32) -> u32 {
    let b = bezout_number(ndim, n);
    if n == 1 {
        return b;
    }

    let sqrt_n = (n as f64).sqrt() as u32;
    b - (1..sqrt_n + 1)
        .filter(|i| n % i == 0)
        .map(|i| 2_u32.pow(ndim).pow(i))
        .sum::<u32>()
}

/// }}}

/// {{{ functions

fn netbrot_compose_fp(brot: &Netbrot, z: &Vector, n: u32) -> Vector {
    match n {
        1 => brot.evaluate(z) - z,
        _ => panic!("Unsupported composition: {}", n),
    }
}

fn netbrot_compose_prime_fp(brot: &Netbrot, z: &Vector, n: u32) -> Matrix {
    let ndim = z.len();

    match n {
        1 => brot.jacobian(z) - Matrix::identity(ndim, ndim),
        _ => panic!("Unsupported composition: {}", n),
    }
}

/// }}}

// {{{ random points

/// Generate a random point in the complex *n*-ball of radius *radius*.
///
/// This function uses the following method to generate a point in the sphere.
///
pub fn generate_random_points_in_ball<R: Rng + ?Sized>(
    rng: &mut R,
    ndim: usize,
    radius: f64,
) -> Vector {
    let normal = Normal::new(0.0, 1.0).unwrap();

    let factor: f64 = rng.gen();
    let components: Vec<f64> = (0..2 * ndim).map(|_| normal.sample(rng)).collect();

    let mut result = Vector::from_iterator(
        ndim,
        (0..ndim).map(|i| c64(components[i], components[i + ndim])),
    );
    result.scale_mut(radius * factor / result.norm());

    result
}

// }}}

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

    let divisors: Vec<u32> = (1..nperiod).filter(|i| nperiod % i == 0).collect();

    // Check if fixed point already exists
    let is_in = fixedpoints.iter().any(|z_j| (z - z_j).norm() < eps);
    if is_in {
        return false;
    }

    // Check if the point is a fixed point
    let is_fp = netbrot_compose_fp(brot, z, nperiod).norm() < eps;
    if !is_fp {
        return false;
    }

    // Check if it's a fixed point of a lower period
    let is_smaller_period = divisors
        .iter()
        .any(|&j| netbrot_compose_fp(brot, z, j).norm() < eps);
    if is_smaller_period {
        return false;
    }

    true
}

// }}}

// {{{ find_fixed_points_by_newton

pub fn find_fixed_points_by_newton(
    brot: &Netbrot,
    nperiod: u32,
    maxit: u32,
    eps: f64,
) -> Vec<Vector> {
    let mut rng = rand::thread_rng();

    let ndim = brot.mat.nrows();
    let npoints = unique_poly_solutions(ndim as u32, nperiod) as usize;
    let radius = brot.escape_radius_squared.sqrt() + 2.0;

    let mut ntries = 0;
    let mut fixedpoints: Vec<Vector> = Vec::with_capacity(npoints);

    let f = |z: &Vector| netbrot_compose_fp(brot, z, nperiod);
    let j = |z: &Vector| netbrot_compose_prime_fp(brot, z, nperiod);
    let solver = NewtonRaphson::new(f, j)
        .with_rtol(eps / 10.0)
        .with_maxit(512);

    while ntries < maxit {
        let z0 = generate_random_points_in_ball(&mut rng, ndim, radius);
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

// {{{ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezout_number() {
        assert_eq!(bezout_number(2, 1), 4);
        assert_eq!(bezout_number(2, 2), 16);
        assert_eq!(bezout_number(2, 3), 64);
        assert_eq!(bezout_number(2, 4), 256);
        assert_eq!(bezout_number(3, 1), 8);
        assert_eq!(bezout_number(3, 2), 64);
    }

    #[test]
    fn test_unique_poly_solutions() {
        assert_eq!(unique_poly_solutions(2, 1), 4);
        assert_eq!(unique_poly_solutions(2, 2), 12);
        assert_eq!(unique_poly_solutions(2, 3), 60);
        assert_eq!(unique_poly_solutions(2, 4), 236);
        assert_eq!(unique_poly_solutions(4, 1), 16);
    }

    #[test]
    fn test_generate_random_points_in_ball() {
        let mut rng = rand::thread_rng();
        let radius = 5.0 * rng.gen::<f64>();

        for ndim in 1..9 {
            for _ in 0..128 {
                let z0 = generate_random_points_in_ball(&mut rng, ndim, radius);
                // println!("ndim {} z0 {} radius {}", ndim, z0.norm(), radius);
                assert!(z0.norm() < radius);
            }
        }
    }
}

// }}}
