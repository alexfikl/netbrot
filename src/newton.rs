// SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use nalgebra::{DMatrix, DVector, RealField, UniformNorm};
use num::complex::Complex;
use num_traits::Float;

const NEWTON_DEFAULT_RTOL: f64 = 1.0e-6;
const NEWTON_DEFAULT_MAXIT: usize = 100;

pub struct NewtonRhapson<T, F, J> {
    /// A $f: \mathbb{C}^n \to \mathbb{C}^n$ function to find the roots of.
    f: F,
    /// A $J: \mathbb{C}^n \to \mathbb{C}^{n \times n}$ function that computes the
    /// Jacobian of *f*.
    j: J,
    /// The relative tolerance (in $x$) at which convergence is assumed.
    rtol: T,
    /// Maximum number of iterations to perform (regardless of *rtol*).
    maxit: usize,
}

#[derive(Debug, PartialEq)]
pub enum NewtonRhapsonError {
    /// Maximum number of iterations was reached and convergence is not achieved.
    MaximumIterationsReached,
    /// Failed to invert the Jacobian.
    BadJacobian,
}

pub type NewtonRhapsonResult<T> = Result<T, NewtonRhapsonError>;

impl<T, F, J> NewtonRhapson<T, F, J>
where
    T: Float + RealField,
    F: Fn(DVector<Complex<T>>) -> DVector<Complex<T>>,
    J: Fn(DVector<Complex<T>>) -> DMatrix<Complex<T>>,
{
    pub fn new(f: F, j: J) -> Self {
        Self {
            f,
            j,
            rtol: T::from(NEWTON_DEFAULT_RTOL).unwrap(),
            maxit: NEWTON_DEFAULT_MAXIT,
        }
    }

    pub fn with_rtol(&mut self, rtol: T) -> &mut Self {
        self.rtol = rtol;
        self
    }

    pub fn with_maxit(&mut self, maxit: usize) -> &mut Self {
        self.maxit = maxit;
        self
    }

    pub fn solve(&self, x0: DVector<Complex<T>>) -> NewtonRhapsonResult<DVector<Complex<T>>> {
        let mut i = 0;
        let mut x = x0.clone();
        let mut err = x.clone().add_scalar(Complex::new(self.rtol, self.rtol));

        while err.apply_norm(&UniformNorm) > self.rtol && i < self.maxit {
            let jac = (self.j)(x.clone());
            let b = (self.f)(x.clone());

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
                Some(residual) => {
                    err.copy_from(&residual);
                    x -= residual;
                }
                None => return Err(NewtonRhapsonError::BadJacobian),
            };

            i += 1;
        }

        if i >= self.maxit {
            Err(NewtonRhapsonError::MaximumIterationsReached)
        } else {
            Ok(x)
        }
    }
}
