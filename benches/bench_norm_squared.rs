// SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use num::complex::{Complex64, c64};

#[inline(never)]
fn escape_iter(z: &mut DVector<Complex64>, maxit: usize) -> f64 {
    let mut result = 0.0;
    for _ in 0..maxit {
        result = z.iter().map(|zi| zi.norm_sqr()).sum();
        z.add_scalar_mut(c64(1.0, 0.0));
    }
    result
}

#[inline(never)]
fn escape_norm_sq(z: &mut DVector<Complex64>, maxit: usize) -> f64 {
    let mut result = 0.0;
    for _ in 0..maxit {
        result = z.norm_squared();
        z.add_scalar_mut(c64(1.0, 0.0));
    }
    result
}

fn bench_iter(c: &mut Criterion) {
    const N: usize = 200;
    const MAXIT: usize = 512;

    let z0 = DVector::<Complex64>::zeros(N);

    c.bench_function("simple_iter", |b| {
        b.iter_batched(
            || z0.clone(),
            |mut z| std::hint::black_box(escape_iter(&mut z, MAXIT)),
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_norm_sq(c: &mut Criterion) {
    const N: usize = 200;
    const MAXIT: usize = 512;

    let z0 = DVector::<Complex64>::zeros(N);

    c.bench_function("simple_norm_sq", |b| {
        b.iter_batched(
            || z0.clone(),
            |mut z| std::hint::black_box(escape_norm_sq(&mut z, MAXIT)),
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, bench_iter, bench_norm_sq);
criterion_main!(benches);
