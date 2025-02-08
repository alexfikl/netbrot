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
use netbrot::iterate::{Netbrot, Vector};
use netbrot::render::{
    render_attractive_fixed_points, render_julia_orbit, render_mandelbrot_orbit, render_period,
    RenderType, Renderer,
};

use nalgebra::DMatrix;
use num::complex::Complex64;
use serde::{Deserialize, Serialize};

use clap::{Parser, ValueHint};
use indicatif::ParallelProgressIterator;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use little_exif::exif_tag::ExifTag;
use little_exif::metadata::Metadata;

// {{{ Command-line parser

#[derive(Parser, Debug)]
#[clap(version, about)]
struct Cli {
    /// The type of render to perform (this mainly has an effect of the colors
    /// and the meaning of the colors)
    #[arg(long, value_enum, default_value = "mandelbrot")]
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

    /// Starting point. When plotting the Mandelbrot set, this corresponds to
    /// z0 and when plotting the Julia set, this corresponds to c.
    #[arg(long, default_values_t = vec![0.0, 0.0], num_args = 2, allow_hyphen_values = true)]
    point: Vec<f64>,

    /// Bounding box for the rendering
    #[arg(long, num_args = 4, allow_hyphen_values = true)]
    bbox: Option<Vec<f64>>,

    /// Period used when looking for attractive fixed points.
    #[arg(long, default_value_t = 1)]
    period: u32,
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
    println!("Iterations:    {}", brot.maxit);
    println!("Escape radius: {}", brot.escape_radius_squared.sqrt());
    match renderer.render_type {
        RenderType::Julia => println!("c:             {}", brot.c),
        _ => {
            if brot.z0.len() <= 5 {
                println!("z0:            {:?}", brot.z0.data.as_vec())
            } else {
                println!("z0 (norm):     {:?}", brot.z0.norm())
            }
        }
    }
}

fn main() {
    let args = Cli::parse();
    let exhibit = read_exhibit(args.exhibit.clone()).unwrap();

    let (upper_left, lower_right) = match args.bbox {
        Some(bbox) => (
            Complex64 {
                re: bbox[0],
                im: bbox[3],
            },
            Complex64 {
                re: bbox[1],
                im: bbox[2],
            },
        ),
        None => (exhibit.upper_left, exhibit.lower_right),
    };

    let renderer = Renderer::new(
        args.resolution,
        // FIXME: make this a proper rectangle
        (upper_left.re, lower_right.re),
        (lower_right.im, upper_left.im),
        args.color,
        args.render,
    );
    let resolution = renderer.resolution;
    let mut pixels = renderer.image();

    let mut brot = Netbrot::new(&exhibit.mat, args.maxit, exhibit.escape_radius);
    match renderer.render_type {
        RenderType::Julia => {
            brot.c = Complex64 {
                re: args.point[0],
                im: args.point[1],
            }
        }
        _ => {
            brot.z0 = Vector::from_element(
                brot.z0.len(),
                Complex64 {
                    re: args.point[0],
                    im: args.point[1],
                },
            )
        }
    }
    display(&renderer, &brot);

    println!("Executing...");
    let now = Instant::now();

    // Scope of slicing up `pixels` into horizontal bands.
    {
        let bands: Vec<(usize, &mut [u8])> =
            pixels.chunks_mut(3 * resolution.0).enumerate().collect();
        let nbands = bands.len() as u64;

        match renderer.render_type {
            RenderType::Julia => {
                bands
                    .into_par_iter()
                    .progress_count(nbands)
                    .for_each(|(i, band)| {
                        let local_renderer = renderer.to_slice(i);
                        render_julia_orbit(&local_renderer, &brot, band);
                    });
            }
            RenderType::Mandelbrot => {
                bands
                    .into_par_iter()
                    .progress_count(nbands)
                    .for_each(|(i, band)| {
                        let local_renderer = renderer.to_slice(i);
                        render_mandelbrot_orbit(&local_renderer, &brot, band);
                    });
            }
            RenderType::Period => {
                bands
                    .into_par_iter()
                    .progress_count(nbands)
                    .for_each(|(i, band)| {
                        let local_renderer = renderer.to_slice(i);
                        render_period(&local_renderer, &brot, band);
                    });
            }
            RenderType::Attractive => {
                bands
                    .into_par_iter()
                    .progress_count(nbands)
                    .for_each(|(i, band)| {
                        let local_renderer = renderer.to_slice(i);
                        render_attractive_fixed_points(&local_renderer, &brot, band, args.period);
                    });
            }
        }
    }

    let elapsed = now.elapsed().as_millis() as f32 / 1000.0;
    println!("Elapsed {}s!", elapsed);

    let outfile = args.outfile.unwrap_or(args.exhibit);
    let filename = Path::new(&outfile).with_extension("png");

    println!("Writing result to '{}'.", filename.display());
    pixels.save(&filename).unwrap();

    println!("Writing metadata to '{}'.", filename.display());
    let mut metadata = Metadata::new();
    metadata.set_tag(ExifTag::ImageDescription(format!(
        "[{}, {}, {}, {}, {}, {}]",
        renderer.bbox.0,
        renderer.bbox.1,
        renderer.bbox.2,
        renderer.bbox.3,
        brot.escape_radius_squared.sqrt(),
        brot.maxit
    )));
    metadata.write_to_file(&filename).unwrap();
}
