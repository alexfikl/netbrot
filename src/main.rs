// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

mod colorschemes;
mod gallery;
mod netbrot;

use netbrot::{pixel_to_point, render_fixed_points, render_orbit, render_period, Netbrot};

use std::time::Instant;

use clap::{Parser, ValueEnum, ValueHint};
use image::RgbImage;
use rayon::prelude::*;

const MAX_ITERATIONS: usize = 256;

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
    /// Fixed points
    Fixed,
}

// }}}

fn main() {
    let args = Cli::parse();

    let color_type = args.color;
    let filename = args.filename;
    println!("Coloring: {:?}", color_type);

    let exhibit = gallery::get_exhibit(gallery::ExhibitId::EXHIBIT_3_DEFAULTS);
    let upper_left = exhibit.upper_left;
    let lower_right = exhibit.lower_right;

    println!(
        "Bounding box: Top left {} Bottom right {}",
        upper_left, lower_right
    );

    let ratio = (lower_right.re - upper_left.re) / (upper_left.im - lower_right.im);
    let resolution = args.resolution as f64;
    let bounds = ((ratio * resolution).round() as usize, resolution as usize);
    println!("Resolution: {}x{}", bounds.0, bounds.1);

    let mut pixels = RgbImage::new(bounds.0 as u32, bounds.1 as u32);

    let brot = Netbrot::new(&exhibit.mat, MAX_ITERATIONS, exhibit.escape_radius);

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
                ColorType::Orbit => {
                    render_orbit(band, &brot, band_bounds, band_upper_left, band_lower_right)
                }
                ColorType::Period => {
                    render_period(band, &brot, band_bounds, band_upper_left, band_lower_right)
                }
                ColorType::Fixed => {
                    render_fixed_points(band, &brot, band_bounds, band_upper_left, band_lower_right)
                }
            }
        });
    }
    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
    println!("Elapsed {}s!", elapsed);

    pixels.save(filename).unwrap();
}
