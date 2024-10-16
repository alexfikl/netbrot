// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

mod colorschemes;
mod fixedpoints;
mod netbrot;
mod newton;
mod render;

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use colorschemes::ColorType;
use netbrot::Netbrot;
use render::{render_orbit, render_period, RenderType};

use nalgebra::DMatrix;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use clap::{Parser, ValueHint};
use image::RgbImage;
use rayon::prelude::*;

// {{{ Command-line parser

#[derive(Parser, Debug)]
#[clap(version, about)]
struct Cli {
    /// The type of render to perform (this mainly has an effect of the colors
    /// and the meaning of the colors)
    #[arg(long, value_enum, default_value = "orbit")]
    render: RenderType,

    /// The color palette to use when rendering
    #[arg(long, value_enum, default_value = "default-palette")]
    color: ColorType,

    /// Resolution of the resulting image (this will be scaled to have the same
    /// ration as the given bounding box)
    #[arg(short, long, default_value_t = 4096)]
    resolution: u32,

    /// Maximum number of iterations before a point is considered in the set
    /// (this will also have an effect on the color intensity)
    #[arg(short, long, default_value_t = 256)]
    maxit: usize,

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
    let render_type = args.render;
    println!("Rendering: {:?}", render_type);

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

    let brot = Netbrot::new(&exhibit.mat, args.maxit, exhibit.escape_radius);
    println!("Escape radius {}", brot.escape_radius_squared.sqrt());

    let result = fixedpoints::find_fixed_points_by_newton(&brot, 512, 2, 1.0e-6);
    for fp in result {
        println!("Fixed points: {}", fp);
    }

    // Scope of slicing up `pixels` into horizontal bands.
    println!("Executing...");
    let now = Instant::now();
    {
        let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(3 * bounds.0).enumerate().collect();

        bands.into_par_iter().for_each(|(i, band)| {
            let top = i;
            let band_bounds = (bounds.0, 1);
            let band_upper_left = render::pixel_to_point(bounds, (0, top), upper_left, lower_right);
            let band_lower_right =
                render::pixel_to_point(bounds, (bounds.0, top + 1), upper_left, lower_right);

            match render_type {
                RenderType::Orbit => render_orbit(
                    band,
                    &brot,
                    band_bounds,
                    band_upper_left,
                    band_lower_right,
                    color_type,
                ),
                RenderType::Period => render_period(
                    band,
                    &brot,
                    band_bounds,
                    band_upper_left,
                    band_lower_right,
                    color_type,
                ),
            }
        });
    }
    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
    println!("Elapsed {}s!", elapsed);

    match args.outfile {
        Some(filename) => {
            println!("Writing result to '{}'.", filename);
            pixels.save(filename).unwrap();
        }
        None => {
            let filename = Path::new(&args.exhibit).with_extension("png");
            println!("Writing result to '{}'.", filename.display());
            pixels.save(filename).unwrap();
        }
    };
}
