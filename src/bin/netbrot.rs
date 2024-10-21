// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

// SPDX-FileCopyrightText: 2016-2024 RustProgramming
// SPDX-License-Identifier: MIT

// NOTE: an initial version of this code was taken from
// https://github.com/ProgrammingRust/mandelbrot/blob/f10fe6859f9fea0d8b2f3d22bb62df8904303de2/src/main.rs

#![warn(rust_2018_idioms)]
#![allow(elided_lifetimes_in_paths)]

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use netbrot::colorschemes::ColorType;
use netbrot::iterate::Netbrot;
use netbrot::render::{
    render_attractive_fixed_points, render_orbit, render_period, RenderType, Renderer,
};

use nalgebra::DMatrix;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use clap::{Parser, ValueHint};
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
    resolution: usize,

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

fn display(renderer: &Renderer, brot: &Netbrot) {
    let bbox = renderer.bbox;

    println!(
        "Resolution:    {}x{}",
        renderer.resolution.0, renderer.resolution.1
    );
    println!(
        "Bounding box:  [{}, {}] x [{}, {}]",
        bbox.0, bbox.1, bbox.2, bbox.3
    );
    println!(
        "Rendering:     {:?} with {:?}",
        renderer.render_type, renderer.color_type
    );

    println!("Netbrot:       {}x{}", brot.mat.nrows(), brot.mat.ncols());
    println!("Escape radius: {}", brot.escape_radius_squared.sqrt());
}

fn main() {
    let args = Cli::parse();
    let exhibit = read_exhibit(args.exhibit.clone()).unwrap();

    let renderer = Renderer::new(
        args.resolution,
        // FIXME: make this a proper rectangle
        (exhibit.upper_left.re, exhibit.lower_right.re),
        (exhibit.lower_right.im, exhibit.upper_left.im),
        args.color,
        args.render,
    );
    let resolution = renderer.resolution;
    let mut pixels = renderer.image();

    let brot = Netbrot::new(&exhibit.mat, args.maxit, exhibit.escape_radius);
    display(&renderer, &brot);

    println!("Executing...");
    let now = Instant::now();

    // Scope of slicing up `pixels` into horizontal bands.
    {
        let bands: Vec<(usize, &mut [u8])> =
            pixels.chunks_mut(3 * resolution.0).enumerate().collect();

        match renderer.render_type {
            RenderType::Orbit => {
                bands.into_par_iter().for_each(|(i, band)| {
                    let local_renderer = renderer.to_slice(i);
                    render_orbit(&local_renderer, &brot, band);
                });
            }
            RenderType::Period => {
                bands.into_par_iter().for_each(|(i, band)| {
                    let local_renderer = renderer.to_slice(i);
                    render_period(&local_renderer, &brot, band);
                });
            }
            RenderType::Attractive => {
                bands.into_par_iter().for_each(|(i, band)| {
                    let local_renderer = renderer.to_slice(i);
                    render_attractive_fixed_points(&local_renderer, &brot, band, 1);
                });
            }
        }
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
