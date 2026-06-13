// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use clap::ValueEnum;
use image::{Rgb, RgbImage};
use num::complex::{Complex64, c64};

use crate::colorschemes::{
    ColorType, get_fixed_point_color, get_period_color, get_smooth_orbit_color,
};
use crate::fixedpoints::{
    FixedPointType, find_fixed_points_by_newton, fixed_point_type, unique_poly_solutions,
};
use crate::iterate::{
    EscapeResult, Netbrot, Vector, netbrot_orbit_escape_1d, netbrot_orbit_escape_2d,
    netbrot_orbit_escape_ndim, netbrot_orbit_period,
};

pub const MAX_PERIODS: usize = 20;
pub const PERIOD_WINDOW: usize = 2 * MAX_PERIODS;

// {{{

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum RenderType {
    /// Plot Julia set: all points $z$ (for a fixed $c$) that do not escape.
    Julia,
    /// Plot Mandelbrot set: all points $c$ (for fixed $z_0$) that do not escape.
    Mandelbrot,
    /// Plot periodicity for orbits that do not escape for a fixed $z_0$.
    Period,
    /// Plot regions of attractive fixed points points.
    Attractive,
}

pub struct Renderer {
    /// Image resolution in pixels `(width x height)`.
    pub resolution: (usize, usize),
    /// Bounding box for the rendered region `(xmin, xmax, ymin, ymax)`.
    pub bbox: (f64, f64, f64, f64),
    /// The coloring type used for rendering.
    pub color_type: ColorType,
    /// The type of rendering.
    pub render_type: RenderType,

    width: f64,
    height: f64,
}

impl Renderer {
    pub fn new(
        resolution: usize,
        xlim: (f64, f64),
        ylim: (f64, f64),
        color_type: ColorType,
        render_type: RenderType,
    ) -> Self {
        let mut ratio = (xlim.1 - xlim.0) / (ylim.1 - ylim.0);
        if ratio.is_nan() || ratio.is_infinite() {
            ratio = 1.0;
        }
        // Clamp ratio to prevent generating ridiculously large values
        let ratio = ratio.abs().clamp(0.01, 10.0);
        let r = resolution as f64;

        Renderer {
            resolution: ((ratio * r).round() as usize, resolution),
            bbox: (xlim.0, xlim.1, ylim.0, ylim.1),
            color_type,
            render_type,

            width: xlim.1 - xlim.0,
            height: ylim.1 - ylim.0,
        }
    }

    /// Width of the rendered image in coordinate space.
    pub fn width(&self) -> f64 {
        self.bbox.1 - self.bbox.0
    }

    /// Height of the rendered image in coordinate space.
    pub fn height(&self) -> f64 {
        self.bbox.3 - self.bbox.2
    }

    /// Create an `RgbImage` for rendering.
    pub fn image(&self) -> RgbImage {
        RgbImage::new(self.resolution.0 as u32, self.resolution.1 as u32)
    }

    /// Translate pixel coordinates to physical point coordinates.
    pub fn pixel_to_point(&self, pixel: (usize, usize)) -> Complex64 {
        let (xmin, _, _, ymax) = self.bbox;

        c64(
            xmin + (pixel.0 as f64) * self.width / (self.resolution.0 as f64),
            ymax - (pixel.1 as f64) * self.height / (self.resolution.1 as f64),
        )
    }

    /// Create a new renderer that only renders an image of size `width x 1` in
    /// the same physical coordinate space as the original.
    ///
    /// *i*: the starting row in pixel space.
    pub fn to_slice(&self, i: usize) -> Self {
        let top = i;
        let resolution = (self.resolution.0, 1);
        let upper_left = self.pixel_to_point((0, top));
        let lower_right = self.pixel_to_point((self.resolution.0, top + 1));

        Renderer {
            resolution,
            bbox: (upper_left.re, lower_right.re, lower_right.im, upper_left.im),
            color_type: self.color_type,
            render_type: self.render_type,

            width: lower_right.re - upper_left.re,
            height: upper_left.im - lower_right.im,
        }
    }
}

// }}}

// {{{ orbit coloring helpers

#[inline]
fn orbit_escape_color(
    color_type: ColorType,
    escape: EscapeResult,
    maxit: usize,
    escape_radius: f64,
) -> Rgb<u8> {
    match escape.iteration {
        None => Rgb([0, 0, 0]),
        Some(n) => get_smooth_orbit_color(color_type, n, escape.z_norm, maxit, escape_radius),
    }
}

#[inline]
fn set_rgb_pixel(pixels: &mut [u8], index: usize, color: Rgb<u8>) {
    pixels[index] = color[0];
    pixels[index + 1] = color[1];
    pixels[index + 2] = color[2];
}

// }}}

// {{{ render Julia orbits

pub fn render_julia_orbit(renderer: &Renderer, brot: &Netbrot, pixels: &mut [u8]) {
    let color_type = renderer.color_type;
    let resolution = renderer.resolution;
    assert!(pixels.len() == 3 * resolution.0 * resolution.1);

    let maxit = brot.maxit;
    let escape_radius = brot.escape_radius_squared.sqrt();
    let escape_r2 = brot.escape_radius_squared;
    let c = brot.c;
    let ndim = brot.z0.len();

    if ndim == 1 {
        let a = brot.mat[(0, 0)];

        for row in 0..resolution.1 {
            for column in 0..resolution.0 {
                let point = renderer.pixel_to_point((column, row));
                let escape = netbrot_orbit_escape_1d(a, point, c, maxit, escape_r2);
                let color = orbit_escape_color(color_type, escape, maxit, escape_radius);
                let index = (row * resolution.0 + column) * 3;
                set_rgb_pixel(pixels, index, color);
            }
        }
        return;
    }

    if ndim == 2 {
        let m = &brot.mat;
        let a00 = m[(0, 0)];
        let a01 = m[(0, 1)];
        let a10 = m[(1, 0)];
        let a11 = m[(1, 1)];

        for row in 0..resolution.1 {
            for column in 0..resolution.0 {
                let point = renderer.pixel_to_point((column, row));
                let escape =
                    netbrot_orbit_escape_2d(a00, a01, a10, a11, point, point, c, maxit, escape_r2);
                let color = orbit_escape_color(color_type, escape, maxit, escape_radius);
                let index = (row * resolution.0 + column) * 3;
                set_rgb_pixel(pixels, index, color);
            }
        }
        return;
    }

    let mut z = brot.z0.clone();
    let mut matz = Vector::zeros(ndim);

    for row in 0..resolution.1 {
        for column in 0..resolution.0 {
            let point = renderer.pixel_to_point((column, row));
            z.fill(point);
            let escape =
                netbrot_orbit_escape_ndim(&brot.mat, &mut z, c, maxit, escape_r2, &mut matz);
            let color = orbit_escape_color(color_type, escape, maxit, escape_radius);
            let index = (row * resolution.0 + column) * 3;
            set_rgb_pixel(pixels, index, color);
        }
    }
}

// }}}

// {{{ render Mandelbrot orbits

pub fn render_mandelbrot_orbit(renderer: &Renderer, brot: &Netbrot, pixels: &mut [u8]) {
    let color_type = renderer.color_type;
    let resolution = renderer.resolution;
    assert!(pixels.len() == 3 * resolution.0 * resolution.1);

    let maxit = brot.maxit;
    let escape_radius = brot.escape_radius_squared.sqrt();
    let escape_r2 = brot.escape_radius_squared;
    let ndim = brot.z0.len();

    if ndim == 1 {
        let a = brot.mat[(0, 0)];
        let z0 = brot.z0[0];

        for row in 0..resolution.1 {
            for column in 0..resolution.0 {
                let c = renderer.pixel_to_point((column, row));
                let escape = netbrot_orbit_escape_1d(a, z0, c, maxit, escape_r2);
                let color = orbit_escape_color(color_type, escape, maxit, escape_radius);
                let index = (row * resolution.0 + column) * 3;
                set_rgb_pixel(pixels, index, color);
            }
        }
        return;
    }

    if ndim == 2 {
        let m = &brot.mat;
        let a00 = m[(0, 0)];
        let a01 = m[(0, 1)];
        let a10 = m[(1, 0)];
        let a11 = m[(1, 1)];
        let z0 = brot.z0[0];
        let z1 = brot.z0[1];

        for row in 0..resolution.1 {
            for column in 0..resolution.0 {
                let c = renderer.pixel_to_point((column, row));
                let escape =
                    netbrot_orbit_escape_2d(a00, a01, a10, a11, z0, z1, c, maxit, escape_r2);
                let color = orbit_escape_color(color_type, escape, maxit, escape_radius);
                let index = (row * resolution.0 + column) * 3;
                set_rgb_pixel(pixels, index, color);
            }
        }
        return;
    }

    let mut z = brot.z0.clone();
    let mut matz = Vector::zeros(ndim);

    for row in 0..resolution.1 {
        for column in 0..resolution.0 {
            z.copy_from(&brot.z0);
            let c = renderer.pixel_to_point((column, row));
            let escape =
                netbrot_orbit_escape_ndim(&brot.mat, &mut z, c, maxit, escape_r2, &mut matz);
            let color = orbit_escape_color(color_type, escape, maxit, escape_radius);
            let index = (row * resolution.0 + column) * 3;
            set_rgb_pixel(pixels, index, color);
        }
    }
}

// }}}

// {{{ render periods

pub fn render_period(renderer: &Renderer, brot: &Netbrot, pixels: &mut [u8]) {
    let color_type = renderer.color_type;
    let resolution = renderer.resolution;
    assert!(pixels.len() == 3 * resolution.0 * resolution.1);

    let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());

    for row in 0..resolution.1 {
        for column in 0..resolution.0 {
            local_brot.c = renderer.pixel_to_point((column, row));
            let color = match netbrot_orbit_period(&local_brot) {
                None => Rgb([255, 255, 255]),
                Some(period) => get_period_color(color_type, period % MAX_PERIODS),
            };

            let index = (row * resolution.0 + column) * 3;
            set_rgb_pixel(pixels, index, color);
        }
    }
}

// }}}

// {{{ render attractive fixed points

pub fn render_attractive_fixed_points(
    renderer: &Renderer,
    brot: &Netbrot,
    pixels: &mut [u8],
    period: u32,
    maxit: u32,
    eps: f64,
) {
    let ndim = brot.z0.len() as u32;
    let color_type = renderer.color_type;
    let resolution = renderer.resolution;
    assert!(pixels.len() == 3 * resolution.0 * resolution.1);

    let mut nfails = 0;
    let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());

    for row in 0..resolution.1 {
        for column in 0..resolution.0 {
            local_brot.c = renderer.pixel_to_point((column, row));
            let fp = find_fixed_points_by_newton(&local_brot, period, maxit, eps);
            let nfps = unique_poly_solutions(ndim, period) as usize;

            let mut color = Rgb([255, 255, 255]);
            if fp.len() < nfps {
                nfails += 1;
            } else {
                color = match fixed_point_type(&local_brot, &fp, period, eps) {
                    FixedPointType::Attractive { eig, stable } => {
                        get_fixed_point_color(color_type, eig, stable)
                    }
                    FixedPointType::Repulsive { .. } => Rgb([255, 255, 255]),
                };
            }

            let index = (row * resolution.0 + column) * 3;
            set_rgb_pixel(pixels, index, color);
        }
    }

    if nfails != 0 {
        println!(
            "Failed to find all roots for {} out of {} points",
            nfails,
            resolution.0 * resolution.1
        );
    }
}

// }}}
