// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use num::complex::c64;
use rand::Rng;

use crate::netbrot::{
    netbrot_repeat, netbrot_repeat_eigenvalues, netbrot_repeat_prime, Netbrot, Vector,
};

/// {{{ find_unique_fixed_points

fn find_unique_fixed_points(
    brot: &Netbrot,
    fixedpoints: &Vec<Vector>,
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

// {{{ find_unique_fixed_points

fn find_fixed_points(brot: &Netbrot, npoints: u32, nperiod: u32, eps: f64) -> Vec<Vector> {
    let mut rng = rand::thread_rng();

    let ndim = brot.mat.nrows();
    let radius = brot.escape_radius_squared.sqrt();

    let mut fixedpoints: Vec<Vector> = Vec::with_capacity(npoints as usize);

    for _ in 0..npoints {
        // FIXME: this should be generated in the sphere, not the cuboid
        let z = Vector::from_vec(
            (0..ndim)
                .map(|_| {
                    c64(
                        rng.gen_range(-radius..radius),
                        rng.gen_range(-radius..radius),
                    )
                })
                .collect(),
        );

        fixedpoints.push(z);
    }

    let indices = find_unique_fixed_points(brot, &fixedpoints, nperiod, eps);
    indices.iter().map(|&i| fixedpoints[i].clone()).collect()
}

// }}}
