// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use std::fmt;

use nalgebra::{DMatrix, DVector, RealField, UniformNorm};
use num::complex::Complex;
use num_traits::Float;

const NEWTON_DEFAULT_RTOL: f64 = 1.0e-6;
const NEWTON_DEFAULT_MAXIT: u32 = 256;

// {{{ Error

#[derive(Eq, Debug, PartialEq)]
pub enum NewtonRaphsonError {
    /// Reached a NaN or an infinite.
    NotFinite,
    /// Maximum number of iterations was reached and convergence is not achieved.
    MaximumIterationsReached,
    /// Failed to invert the Jacobian.
    BadJacobian,
}

impl NewtonRaphsonError {
    fn as_str(&self) -> &'static str {
        match *self {
            NewtonRaphsonError::NotFinite => "Function evaluation resulted in a NaN",
            NewtonRaphsonError::MaximumIterationsReached => {
                "Maximum number of iterations reached, but tolerance not achieved"
            }
            NewtonRaphsonError::BadJacobian => "Jacobian is singular (failed LU decomposition)",
        }
    }
}

impl fmt::Display for NewtonRaphsonError {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.write_str(self.as_str())
    }
}

// }}}

// {{{ NewtonRaphson

pub struct NewtonRaphson<T, F, J> {
    /// A $f: \mathbb{C}^n \to \mathbb{C}^n$ function to find the roots of.
    f: F,
    /// A $J: \mathbb{C}^n \to \mathbb{C}^{n \times n}$ function that computes the
    /// Jacobian of *f*.
    j: J,
    /// The relative tolerance (in $x$) at which convergence is assumed.
    rtol: T,
    /// Maximum number of iterations to perform (regardless of *rtol*).
    maxit: u32,
}

pub type Vector<T> = DVector<Complex<T>>;
pub type Matrix<T> = DMatrix<Complex<T>>;

#[allow(dead_code)]
pub struct NewtonRaphsonResult<T> {
    /// Solution vector.
    pub x: Vector<T>,
    /// Number of iterations necessary to reach solution.
    pub iteration: u32,
}

impl<T, F, J> NewtonRaphson<T, F, J>
where
    T: Float + RealField,
    F: Fn(&Vector<T>) -> Vector<T>,
    J: Fn(&Vector<T>) -> Matrix<T>,
{
    pub fn new(f: F, j: J) -> Self {
        Self {
            f,
            j,
            rtol: T::from(NEWTON_DEFAULT_RTOL).unwrap(),
            maxit: NEWTON_DEFAULT_MAXIT,
        }
    }

    pub fn with_rtol(mut self, rtol: T) -> Self {
        self.rtol = rtol;
        self
    }

    pub fn with_maxit(mut self, maxit: u32) -> Self {
        self.maxit = maxit;
        self
    }

    pub fn solve(&self, x0: Vector<T>) -> Result<NewtonRaphsonResult<T>, NewtonRaphsonError> {
        let mut i = 0_u32;
        let mut x = x0.clone();

        while i < self.maxit {
            let b = (self.f)(&x);
            let bnorm = b.apply_norm(&UniformNorm);
            if !bnorm.is_finite() {
                return Err(NewtonRaphsonError::NotFinite);
            }

            if bnorm < self.rtol {
                break;
            }

            let jac = (self.j)(&x);

            // Solve the standard Newton equation
            //          x_{n + 1} = x_n - J^{-1}(x_n) f(x_n)
            //      =>  J(x_n) (x_n - x_{n + 1}) = f(x_n)
            //
            // which gives a standard `A r = b` type system with the Jacobian.
            // We solve that with LU because it's nicely implemented in nalgebra.
            // Then we can set
            //          r = x_n - x_{n + 1}
            //      =>  x_{n + 1} = x_n - r
            //
            // where `r` also just denotes the error between the two iterates.
            let lu_decomp = jac.lu();
            match lu_decomp.solve(&b) {
                Some(residual) => x -= residual,
                None => return Err(NewtonRaphsonError::BadJacobian),
            };

            i += 1;
        }

        if i >= self.maxit {
            Err(NewtonRaphsonError::MaximumIterationsReached)
        } else {
            Ok(NewtonRaphsonResult { x, iteration: i })
        }
    }
}

// }}}

// {{{ tests

#[cfg(test)]
mod tests {
    use super::*;

    use nalgebra::{dmatrix, dvector};
    use num::complex::c64;

    use crate::netbrot::{Matrix, Vector};

    fn f_wikipedia(z: &Vector) -> Vector {
        dvector![
            5.0 * z[0].powi(2) + z[0] * z[1].powi(2) + (2.0 * z[1]).sin().powi(2) - 2.0,
            (2.0 * z[0] - z[1]).exp() + 4.0 * z[1] - 3.0,
        ]
    }

    fn j_wikipedia(z: &Vector) -> Matrix {
        dmatrix![
            10.0 * z[0] + z[1].powi(2),
            2.0 * z[0] * z[1] + 4.0 * (2.0 * z[1]).sin() * (2.0 * z[1]).cos();
            2.0 * (2.0 * z[0] - z[1]).exp(),
            -(2.0 * z[0] - z[1]).exp() + 4.0
        ]
    }

    #[test]
    fn test_wikipedia_example() {
        // https://en.wikipedia.org/wiki/Newton%27s_method#Example
        let z0 = dvector![c64(1.0, 0.0), c64(1.0, 0.0)];
        let newton = NewtonRaphson::new(f_wikipedia, j_wikipedia);

        match newton.solve(z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                let fz = f_wikipedia(&z).apply_norm(&UniformNorm);
                println!("zstar {}f(z) {:e} rtol {:e}", z, fz, NEWTON_DEFAULT_RTOL);
                assert!(fz < NEWTON_DEFAULT_RTOL);
            }
            Err(_) => unreachable!(),
        };
    }

    #[test]
    fn test_square() {
        // Solve x^2 - 2.0 == 0 => x = sqrt(2)
        let f = |z: &Vector| dvector![z[0] * z[0] - 2.0];
        let j = |z: &Vector| dmatrix![2.0 * z[0]];

        let z0 = dvector![c64(1.0, 0.0)];
        let newton = NewtonRaphson::new(f, j);

        match newton.solve(z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                let fz = f(&z).apply_norm(&UniformNorm);
                println!("zstar {}f(z) {:e} rtol {:e}", z, fz, NEWTON_DEFAULT_RTOL);
                assert!(fz < NEWTON_DEFAULT_RTOL);
                assert!((z[0].re - 2.0.sqrt()).abs() < NEWTON_DEFAULT_RTOL);
            }
            Err(_) => unreachable!(),
        };
    }

    #[test]
    fn test_high_derivative() {
        let f = |z: &Vector| dvector![1e9 * z[0].powi(9) - 1.0];
        let j = |z: &Vector| dmatrix![9e9 * z[0].powi(8)];

        let z0 = dvector![c64(0.15, 0.0)];
        let newton = NewtonRaphson::new(f, j).with_maxit(16);

        match newton.solve(z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                let fz = f(&z).apply_norm(&UniformNorm);
                println!("zstar {}f(z) {:e} rtol {:e}", z, fz, NEWTON_DEFAULT_RTOL);
                assert!(fz < NEWTON_DEFAULT_RTOL);
                assert!((z[0].re - 0.1).abs() < NEWTON_DEFAULT_RTOL);
            }
            Err(_) => unreachable!(),
        };
    }

    fn f_broyden1965_case5(z: &Vector, alpha: f64, beta: f64) -> Vector {
        dvector![
            -(3.0 + alpha * z[0]) * z[0] + 2.0 * z[1] - beta,
            z[0] - (3.0 + alpha * z[1]) * z[1] + 2.0 * z[2] - beta,
            z[1] - (3.0 + alpha * z[2]) * z[2] + 2.0 * z[3] - beta,
            z[2] - (3.0 + alpha * z[3]) * z[3] + 2.0 * z[4] - beta,
            z[3] - (3.0 + alpha * z[4]) * z[4] - beta,
        ]
    }

    fn j_broyden1965_case5(z: &Vector, alpha: f64, _beta: f64) -> Matrix {
        dmatrix![
            // row 0
            -(3.0 + 2.0 * alpha * z[0]),
            c64(2.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0);
            // row 1
            c64(1.0, 0.0),
            -(3.0 + 2.0 * alpha * z[1]),
            c64(2.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0);
            // row 2
            c64(0.0, 0.0),
            c64(1.0, 0.0),
            -(3.0 + 2.0 * alpha * z[2]),
            c64(2.0, 0.0),
            c64(0.0, 0.0);
            // row 3
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(1.0, 0.0),
            -(3.0 + 2.0 * alpha * z[3]),
            c64(2.0, 0.0);
            // row 4
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(0.0, 0.0),
            c64(1.0, 0.0),
            -(3.0 + 2.0 * alpha * z[4])
        ]
    }

    #[test]
    fn test_broyden1965_case5() {
        let f = |z: &Vector| f_broyden1965_case5(z, -0.1, 1.0);
        let j = |z: &Vector| j_broyden1965_case5(z, -0.1, 1.0);

        let z0 = Vector::from_element(5, c64(-1.0, 0.0));
        let newton = NewtonRaphson::new(f, j);

        match newton.solve(z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                let fz = f(&z).apply_norm(&UniformNorm);
                println!("zstar {}f(z) {:e} rtol {:e}", z, fz, NEWTON_DEFAULT_RTOL);
                assert!(fz < NEWTON_DEFAULT_RTOL);
            }
            Err(_) => unreachable!(),
        };
    }

    #[test]
    fn test_broyden1965_case6() {
        let f = |z: &Vector| f_broyden1965_case5(z, -0.5, 1.0);
        let j = |z: &Vector| j_broyden1965_case5(z, -0.5, 1.0);

        let z0 = Vector::from_element(5, c64(-1.0, 0.0));
        let newton = NewtonRaphson::new(f, j);

        match newton.solve(z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                let fz = f(&z).apply_norm(&UniformNorm);
                println!("zstar {}f(z) {:e} rtol {:e}", z, fz, NEWTON_DEFAULT_RTOL);
                assert!(fz < NEWTON_DEFAULT_RTOL);
            }
            Err(_) => unreachable!(),
        };
    }

    fn f_broyden1965_case9(z: &Vector) -> Vector {
        dvector![10.0 * (z[1] - z[0] * z[0]), 1.0 - z[0]]
    }

    fn j_broyden1965_case9(z: &Vector) -> Matrix {
        dmatrix![-20.0 * z[0], c64(10.0, 0.0); c64(-1.0, 0.0), c64(0.0, 0.0)]
    }

    #[test]
    fn test_broyden1965_case9() {
        let f = |z: &Vector| f_broyden1965_case9(z);
        let j = |z: &Vector| j_broyden1965_case9(z);

        let z0 = dvector![c64(-1.2, 0.0), c64(1.0, 0.0)];
        let newton = NewtonRaphson::new(f, j).with_maxit(512);

        match newton.solve(z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: _ }) => {
                let fz = f(&z).apply_norm(&UniformNorm);
                println!("zstar {}f(z) {:e} rtol {:e}", z, fz, NEWTON_DEFAULT_RTOL);
                assert!(fz < NEWTON_DEFAULT_RTOL);
            }
            Err(e) => panic!("{:?}", e),
        };
    }

    fn f_broyden1965_case10(z: &Vector) -> Vector {
        dvector![
            -13.0 + z[0] + ((5.0 - z[1]) * z[1] - 2.0) * z[1],
            -29.0 + z[0] + ((1.0 + z[1]) * z[1] - 14.0) * z[1],
        ]
    }

    fn j_broyden1965_case10(z: &Vector) -> Matrix {
        dmatrix![
            c64(1.0, 0.0),
            -2.0 + 10.0 * z[1] - 3.0 * z[1] * z[1];
            c64(1.0, 0.0),
            -14.0 + 2.0 * z[1] + 3.0 * z[1] * z[1]
        ]
    }

    #[test]
    fn test_broyden1965_case10() {
        let f = |z: &Vector| f_broyden1965_case10(z);
        let j = |z: &Vector| j_broyden1965_case10(z);

        let z0 = dvector![c64(15.0, 0.0), c64(-2.0, 0.0)];
        let newton = NewtonRaphson::new(f, j);

        match newton.solve(z0) {
            Ok(NewtonRaphsonResult { x: z, iteration: n }) => {
                let fz = f(&z).apply_norm(&UniformNorm);
                println!(
                    "[{}] zstar {}f(z) {:e} rtol {:e}",
                    n, z, fz, NEWTON_DEFAULT_RTOL
                );
                assert!(fz < NEWTON_DEFAULT_RTOL);
            }
            Err(_) => unreachable!(),
        };
    }
}

// }}}
