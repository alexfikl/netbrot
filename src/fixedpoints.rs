// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::c64;
use rand::Rng;
use rand_distr::{Distribution, Normal};

use crate::iterate::{Matrix, Netbrot, Vector};
use crate::newton::{NewtonRaphson, NewtonRaphsonResult};

// {{{ polynomial solutions

pub fn bezout_number(ndim: u32, n: u32) -> u32 {
    2_u32.pow(ndim).pow(n)
}

pub fn unique_poly_solutions(ndim: u32, n: u32) -> u32 {
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

// }}}

// {{{ functions

fn netbrot_compose(brot: &Netbrot, z: &Vector, n: u32) -> Vector {
    match n {
        1 => brot.evaluate(z),
        n => {
            let mut result = brot.evaluate(z);
            for _ in 0..n - 1 {
                result = brot.evaluate(&result);
            }

            result
        }
    }
}

fn netbrot_compose_fp(brot: &Netbrot, z: &Vector, n: u32) -> Vector {
    let mut result = netbrot_compose(brot, z, n);
    result -= z;

    result
}

fn netbrot_compose_prime(brot: &Netbrot, z: &Vector, n: u32) -> Matrix {
    match n {
        1 => brot.jacobian(z),
        n => {
            let mut result = brot.jacobian(z);
            let mut fz = z.clone_owned();
            for _ in 0..n - 1 {
                fz = brot.evaluate(&fz);
                result = brot.jacobian(&fz) * result;
            }

            result
        }
    }
}

fn netbrot_compose_prime_fp(brot: &Netbrot, z: &Vector, n: u32) -> Matrix {
    let ndim = z.len();
    let mut result = netbrot_compose_prime(brot, z, n);
    for i in 0..ndim {
        result[(i, i)] -= 1.0;
    }

    result
}

// }}}

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

#[allow(dead_code)]
pub fn generate_random_vector<R: Rng + ?Sized>(rng: &mut R, ndim: usize) -> Vector {
    let components: Vec<f64> = (0..2 * ndim).map(|_| rng.gen()).collect();

    Vector::from_iterator(
        ndim,
        (0..ndim).map(|i| c64(components[i], components[i + ndim])),
    )
}

// }}}

// {{{ find_unique_fixed_points

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
    let divisors: Vec<u32> = (1..nperiod).filter(|i| nperiod % i == 0).collect();
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

#[allow(dead_code)]
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

    while ntries < maxit && fixedpoints.len() < npoints {
        let z0 = generate_random_points_in_ball(&mut rng, ndim, radius);

        match solver.solve(&z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                if is_unique_fixed_point(&fixedpoints, &z, brot, nperiod, eps) {
                    fixedpoints.push(z)
                }
            }
            Err(_) => continue,
        }

        // TODO: only increment on successful attempts?
        ntries += 1;
    }

    fixedpoints
}

// }}}

// {{{ check attractiveness

pub enum FixedPointType {
    Attractive(f64),
    Repulsive(f64),
}

pub fn fixed_point_type(brot: &Netbrot, fixedpoints: &Vec<Vector>, period: u32) -> FixedPointType {
    let mut lambda_max = 0.0_f64;

    for zstar in fixedpoints {
        let jac = netbrot_compose_prime(brot, zstar, period);
        let lambdas = jac.eigenvalues().unwrap();
        let lambda_max_i = lambdas.iter().fold(0.0, |acc, z| z.norm().max(acc));

        if lambda_max_i < 1.01 {
            return FixedPointType::Attractive(lambda_max_i);
        }
        lambda_max = lambda_max.max(lambda_max_i);
    }

    FixedPointType::Repulsive(lambda_max)
}

// }}}

// {{{ tests

#[cfg(test)]
mod tests {
    use super::*;

    use nalgebra::{dmatrix, dvector};

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

    #[test]
    fn test_compose_fp() {
        let ndim = 2;
        let maxit = 512;
        let escape_radius = 5.0;

        let mat = dmatrix![c64(1.0, 0.0), c64(0.8, 0.0); c64(1.0, 0.0), c64(-0.5, 0.0)];
        let brot = Netbrot::new(&mat, maxit, escape_radius);

        let mut rng = rand::thread_rng();

        // period = 1
        for _ in 0..32 {
            let z = generate_random_points_in_ball(&mut rng, ndim, escape_radius);

            let f0 = netbrot_compose_fp(&brot, &z, 1);
            let f1 = brot.evaluate(&z) - &z;
            assert!((&f0 - &f1).norm() < 1.0e-15 * f0.norm());

            let j0 = netbrot_compose_prime_fp(&brot, &z, 1);
            let j1 = brot.jacobian(&z) - Matrix::identity(ndim, ndim);
            assert!((&j0 - &j1).norm() < 1.0e-15 * j0.norm());
        }

        // period = 2
        for _ in 0..32 {
            let z = generate_random_points_in_ball(&mut rng, ndim, escape_radius);

            let f0 = netbrot_compose_fp(&brot, &z, 2);
            let f1 = brot.evaluate(&z);
            let f2 = brot.evaluate(&f1) - &z;
            assert!((&f0 - &f2).norm() < 1.0e-15 * f0.norm());

            let j0 = netbrot_compose_prime_fp(&brot, &z, 2);
            let j1 = brot.jacobian(&f1);
            let j2 = j1 * brot.jacobian(&z) - Matrix::identity(ndim, ndim);
            assert!((&j0 - &j2).norm() < 1.0e-15 * j0.norm());
        }

        // repeated composition + chain rule
        for _ in 0..32 {
            let z = generate_random_points_in_ball(&mut rng, ndim, escape_radius);

            let f0 = netbrot_compose(&brot, &z, 5);
            let f1 = netbrot_compose(&brot, &netbrot_compose(&brot, &z, 3), 2);
            assert!((&f0 - &f1).norm() < 1.0e-15 * f0.norm());

            let j0 = netbrot_compose_prime(&brot, &z, 5);
            let f3 = netbrot_compose(&brot, &z, 3);
            let j1 = netbrot_compose_prime(&brot, &f3, 2) * netbrot_compose_prime(&brot, &z, 3);
            assert!((&j0 - &j1).norm() < 1.0e-15 * j0.norm());
        }
    }

    #[test]
    fn test_find_fixed_points_by_newton() {
        let ndim = 2_usize;
        let maxit = 512_usize;
        let escape_radius = 3.4742662001265163_f64;
        let eps = 1.0e-8_f64;

        let mat = dmatrix![c64(1.0, 0.0), c64(0.8, 0.0); c64(1.0, 0.0), c64(-0.5, 0.0)];
        let brot = Netbrot::new(&mat, maxit, escape_radius);

        // NOTE: Obtained from Mathematica to 15 digits
        let fp = [
            dvector![c64(0.0, 0.0), c64(0.0, 0.0)],
            dvector![c64(0.585929683557274, 0.0), c64(0.224413444208041, 0.0)],
            dvector![
                c64(-0.618408628760886, -0.002964957123238),
                c64(0.775367242393021, -0.979283648571905)
            ],
            dvector![
                c64(-0.618408628760886, 0.002964957123238),
                c64(0.775367242393021, 0.979283648571905)
            ],
        ];

        // check that the roots are actually roots
        for z in fp.iter() {
            assert!(z.norm() < escape_radius);
            assert!(netbrot_compose_fp(&brot, z, 1).norm() < 1.0e-14);
        }

        // check that a small perturbation of the roots works
        let f = |z: &Vector| netbrot_compose_fp(&brot, z, 1);
        let j = |z: &Vector| netbrot_compose_prime_fp(&brot, z, 1);
        let solver = NewtonRaphson::new(f, j)
            .with_rtol(eps / 10.0)
            .with_maxit(512);

        let mut rng = rand::thread_rng();
        for z in fp.iter() {
            let zeps = z + generate_random_vector(&mut rng, ndim).scale(0.1);

            match solver.solve(&zeps) {
                Ok(NewtonRaphsonResult {
                    x: zstar,
                    iteration: _,
                }) => {
                    // println!("z {} zeps {} zstar {}", z, zeps, zstar);
                    assert!((zstar - z).norm() < 10.0 * eps);
                }
                Err(_) => unreachable!(),
            }
        }

        // check the main routine with random points
        let fp_est = find_fixed_points_by_newton(&brot, 1, 1024, eps);
        for z in fp_est.iter() {
            assert!(z.norm() < escape_radius);
            assert!(netbrot_compose_fp(&brot, z, 1).norm() < eps);
        }

        for z_est in fp_est {
            let found = fp.iter().any(|z_j| (&z_est - z_j).norm() < eps);
            assert!(found);
        }
    }
}

// }}}
