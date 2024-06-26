// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

mod colorschemes;
use colorschemes::{get_period_color, get_smooth_orbit_color};

mod mandelbrot;
use mandelbrot::{netbrot_orbit_escape, netbrot_orbit_period, MAX_PERIODS};

use std::time::Instant;

use clap::{Parser, ValueEnum, ValueHint};
use image::{Rgb, RgbImage};
use nalgebra::{matrix, vector, SMatrix, SVector};
use num::Complex;
use rayon::prelude::*;

const MAX_ITERATIONS: usize = 256;

macro_rules! c64 {
    ($re: literal) => {
        Complex { re: $re, im: 0.0 }
    };
}

// {{{ Rendering

fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) -> Complex<f64> {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );
    Complex {
        // Why subtraction here? pixel.1 increases as we go down,
        re: upper_left.re + (pixel.0 as f64) * width / (bounds.0 as f64),
        // but the imaginary component increases as we go up.
        im: upper_left.im - (pixel.1 as f64) * height / (bounds.1 as f64),
    }
}

fn render_orbit<const D: usize>(
    pixels: &mut [u8],
    mat: SMatrix<Complex<f64>, D, D>,
    z0: SVector<Complex<f64>, D>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit_escape(point, mat, z0, MAX_ITERATIONS) {
                (None, _) => Rgb([0, 0, 0]),
                (Some(n), z) => get_smooth_orbit_color(n, z.norm(), MAX_ITERATIONS),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index + 0] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

fn render_period<const D: usize>(
    pixels: &mut [u8],
    mat: SMatrix<Complex<f64>, D, D>,
    z0: SVector<Complex<f64>, D>,
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit_period(point, mat, z0, MAX_ITERATIONS) {
                None => Rgb([255, 255, 255]),
                Some(period) => get_period_color(period, MAX_PERIODS, 3),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index + 0] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

// }}}

// {{{ Command-line parser

#[derive(Parser, Debug)]
#[clap(version, about)]
struct Cli {
    /// If given, plot periods instead of orbits
    #[arg(short, long, value_enum, default_value = "orbit")]
    color: ColorType,

    /// Resolution of the resulting image
    #[arg(short, long, default_value_t = 8000)]
    resolution: u32,

    /// Output file name
    #[arg(last = true, value_hint = ValueHint::FilePath)]
    filename: String,
}

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
enum ColorType {
    /// Plot orbits.
    Orbit,
    /// Plot periodicity for orbits that do not escape.
    Period,
}

// }}}

fn main() {
    let args = Cli::parse();

    let color_type = args.color;
    let filename = args.filename;
    println!("Coloring: {:?}", color_type);

    // Full 2x2 brot interval
    // let upper_left = Complex { re: -0.9, im: 0.6 };
    // let lower_right = Complex { re: 0.4, im: -0.6 };

    // Full 3x3 brot interval
    let upper_left = Complex {
        re: -1.25,
        im: 0.75,
    };
    let lower_right = Complex { re: 0.5, im: -0.75 };

    // Baby 3x3 brot interval
    // let upper_left = Complex {
    //     re: -1.025,
    //     im: 0.025,
    // };
    // let lower_right = Complex {
    //     re: -0.975,
    //     im: -0.025,
    // };
    println!(
        "Bounding box: Top left {} Bottom right {}",
        upper_left, lower_right
    );

    let ratio = (lower_right.re - upper_left.re) / (upper_left.im - lower_right.im);
    let resolution = args.resolution as f64;
    let bounds = ((ratio * resolution).round() as usize, resolution as usize);
    println!("Resolution: {}x{}", bounds.0, bounds.1);

    let mut pixels = RgbImage::new(bounds.0 as u32, bounds.1 as u32);

    let z0 = vector![c64!(0.0), c64!(0.0), c64!(0.0)];
    let mat = matrix![
        c64!(1.0), c64!(0.0), c64!(0.0);
        c64!(-1.0), c64!(1.0), c64!(0.0);
        c64!(1.0), c64!(1.0), c64!(-1.0);
    ];
    // let z0 = vector![c64!(0.0), c64!(0.0)];
    // let mat = matrix![
    //     c64!(1.0), c64!(0.8);
    //     c64!(1.0), c64!(-0.5);
    // ];
    // let mat = matrix![
    //     c64!(1.0), c64!(1.0);
    //     c64!(0.0), c64!(1.0);
    // ];

    // Scope of slicing up `pixels` into horizontal bands.
    println!("Executing...");
    let now = Instant::now();
    {
        let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(3 * bounds.0).enumerate().collect();

        bands.into_par_iter().for_each(|(i, band)| {
            let top = i;
            let band_bounds = (bounds.0, 1);
            let band_upper_left = pixel_to_point(bounds, (0, top), upper_left, lower_right);
            let band_lower_right =
                pixel_to_point(bounds, (bounds.0, top + 1), upper_left, lower_right);

            match color_type {
                ColorType::Orbit => render_orbit(
                    band,
                    mat,
                    z0,
                    band_bounds,
                    band_upper_left,
                    band_lower_right,
                ),
                ColorType::Period => render_period(
                    band,
                    mat,
                    z0,
                    band_bounds,
                    band_upper_left,
                    band_lower_right,
                ),
            }
        });
    }
    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
    println!("Elapsed {}s!", elapsed);

    pixels.save(filename).unwrap();
}
