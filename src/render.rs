// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use clap::ValueEnum;
use image::Rgb;
use num::complex::{c64, Complex64};

use crate::colorschemes::{get_period_color, get_smooth_orbit_color, ColorType};
use crate::iterate::{netbrot_orbit, netbrot_orbit_period, EscapeResult, Netbrot};

pub const MAX_PERIODS: usize = 20;
pub const PERIOD_WINDOW: usize = 2 * MAX_PERIODS;

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderType {
    /// Plot orbits.
    Orbit,
    /// Plot periodicity for orbits that do not escape.
    Period,
    // Fixed points
    // Fixed,
}

/// Translate pixel coordinates to physical point coordinates.
///
/// *bounds*: width and height of the image.
/// *upper_left*, *lower_left*: bounding box of the domain.
pub fn pixel_to_point(
    bounds: (usize, usize),
    pixel: (usize, usize),
    upper_left: Complex64,
    lower_right: Complex64,
) -> Complex64 {
    let (width, height) = (
        lower_right.re - upper_left.re,
        upper_left.im - lower_right.im,
    );
    c64(
        // Why subtraction here? pixel.1 increases as we go down,
        upper_left.re + (pixel.0 as f64) * width / (bounds.0 as f64),
        // but the imaginary component increases as we go up.
        upper_left.im - (pixel.1 as f64) * height / (bounds.1 as f64),
    )
}

// {{{ render orbits

pub fn render_orbit(
    pixels: &mut [u8],
    brot: &Netbrot,
    bounds: (usize, usize),
    upper_left: Complex64,
    lower_right: Complex64,
    color: ColorType,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);
    let maxit = brot.maxit;
    let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            local_brot.c = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit(&local_brot) {
                EscapeResult {
                    iteration: None,
                    z: _,
                } => Rgb([0, 0, 0]),
                EscapeResult {
                    iteration: Some(n),
                    z,
                } => get_smooth_orbit_color(color, n, z.norm(), maxit),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

// }}}

// {{{ render periods

pub fn render_period(
    pixels: &mut [u8],
    brot: &Netbrot,
    bounds: (usize, usize),
    upper_left: Complex64,
    lower_right: Complex64,
    color: ColorType,
) {
    assert!(pixels.len() == 3 * bounds.0 * bounds.1);
    let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());

    for row in 0..bounds.1 {
        for column in 0..bounds.0 {
            local_brot.c = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            let color = match netbrot_orbit_period(&local_brot) {
                None => Rgb([255, 255, 255]),
                Some(period) => get_period_color(color, period % MAX_PERIODS),
            };

            let index = row * bounds.0 + 3 * column;
            pixels[index] = color[0];
            pixels[index + 1] = color[1];
            pixels[index + 2] = color[2];
        }
    }
}

// }}}
