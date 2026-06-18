// SPDX-FileCopyrightText: 2026 Alexandru Fikl <alexfikl@gmail.com>
// SPDX-License-Identifier: MIT

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::DVector;
use num::complex::{Complex64, c64};

/// Original: `z[i] = matz[i] * matz[i] + c` via manual loop
#[inline(never)]
fn square_add_loop(
    z: &mut DVector<Complex64>,
    matz: &DVector<Complex64>,
    c: Complex64,
    maxit: usize,
) {
    for _ in 0..maxit {
        for (zi, wi) in z.iter_mut().zip(matz.iter()) {
            *zi = wi * wi + c;
        }
    }
}

#[inline(never)]
fn square_add_nalgebra_inplace(
    z: &mut DVector<Complex64>,
    matz: &DVector<Complex64>,
    c: Complex64,
    maxit: usize,
) {
    for _ in 0..maxit {
        z.copy_from(matz);
        z.component_mul_assign(matz);
        z.add_scalar_mut(c);
    }
}

#[inline(never)]
fn square_add_nalgebra_zip_apply(
    z: &mut DVector<Complex64>,
    matz: &DVector<Complex64>,
    c: Complex64,
    maxit: usize,
) {
    for _ in 0..maxit {
        z.zip_apply(matz, |zi, wi| *zi = wi * wi + c);
    }
}

fn bench_square_add_loop(cr: &mut Criterion) {
    const N: usize = 200;
    const MAXIT: usize = 512;

    let z0 = DVector::<Complex64>::zeros(N);
    let matz0 = DVector::<Complex64>::zeros(N);
    let c = c64(1.0, 0.0);

    cr.bench_function("square_add_loop", |b| {
        b.iter_batched(
            || (z0.clone(), matz0.clone()),
            |(mut z, matz)| {
                square_add_loop(&mut z, &matz, c, MAXIT);
                std::hint::black_box(());
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_square_add_nalgebra_inplace(cr: &mut Criterion) {
    const N: usize = 200;
    const MAXIT: usize = 512;

    let z0 = DVector::<Complex64>::zeros(N);
    let matz0 = DVector::<Complex64>::zeros(N);
    let c = c64(1.0, 0.0);

    cr.bench_function("square_add_nalgebra_inplace", |b| {
        b.iter_batched(
            || (z0.clone(), matz0.clone()),
            |(mut z, matz)| {
                square_add_nalgebra_inplace(&mut z, &matz, c, MAXIT);
                std::hint::black_box(());
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn bench_square_add_nalgebra_zip_apply(cr: &mut Criterion) {
    const N: usize = 200;
    const MAXIT: usize = 512;

    let z0 = DVector::<Complex64>::zeros(N);
    let matz0 = DVector::<Complex64>::zeros(N);
    let c = c64(1.0, 0.0);

    cr.bench_function("square_add_nalgebra_zip_apply", |b| {
        b.iter_batched(
            || (z0.clone(), matz0.clone()),
            |(mut z, matz)| {
                square_add_nalgebra_zip_apply(&mut z, &matz, c, MAXIT);
                std::hint::black_box(());
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

criterion_group!(
    benches,
    bench_square_add_loop,
    bench_square_add_nalgebra_inplace,
    bench_square_add_nalgebra_zip_apply,
);
criterion_main!(benches);
