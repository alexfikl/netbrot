// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

mod colorschemes;
mod netbrot;

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use netbrot::{pixel_to_point, render_fixed_points, render_orbit, render_period, Netbrot};

use nalgebra::DMatrix;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use clap::{Parser, ValueEnum, ValueHint};
use image::RgbImage;
use rayon::prelude::*;

const MAX_ITERATIONS: usize = 256;

// {{{ Command-line parser

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
enum ColorType {
    /// Plot orbits.
    Orbit,
    /// Plot periodicity for orbits that do not escape.
    Period,
    /// Fixed points
    Fixed,
}

#[derive(Parser, Debug)]
#[clap(version, about)]
struct Cli {
    /// If given, plot periods instead of orbits
    #[arg(short, long, value_enum, default_value = "orbit")]
    color: ColorType,

    /// Resolution of the resulting image
    #[arg(short, long, default_value_t = 8000)]
    resolution: u32,

    /// Input file name containing the exhibit to render
    #[arg(value_hint = ValueHint::FilePath)]
    exhibit: String,

    /// Output file name
    #[arg(short, long, value_hint = ValueHint::FilePath)]
    outfile: Option<String>,
}

// {{ exhibits

#[derive(Serialize, Deserialize)]
pub struct Exhibit {
    /// Matrix used in the iteration.
    pub mat: DMatrix<Complex64>,
    /// Escape radius for this matrix.
    pub escape_radius: f64,
    /// Bounding box for the points.
    pub upper_left: Complex64,
    pub lower_right: Complex64,
}

fn read_exhibit(filename: String) -> Result<Exhibit, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);

    let exhibit = serde_json::from_reader(reader)?;

    Ok(exhibit)
}

// }}}

fn main() {
    let args = Cli::parse();

    let color_type = args.color;
    println!("Coloring: {:?}", color_type);

    let exhibit = read_exhibit(args.exhibit.clone()).unwrap();
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

    match args.outfile {
        Some(filename) => pixels.save(filename).unwrap(),
        None => {
            let filename = Path::new(&args.exhibit).with_extension("png");
            pixels.save(filename).unwrap();
        }
    };
}
