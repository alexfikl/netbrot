Netbrot
=======

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)
[![Build Status](https://github.com/inducer/netbrot/actions/workflows/ci.yml/badge.svg)](https://github.com/inducer/netbrot/actions/workflows/ci.yml)
[![Zenodo repository](https://zenodo.org/badge/DOI/10.5281/zenodo.15573719.svg)](https://doi.org/10.5281/zenodo.15573719)

This repository contains some experiments for vector Mandelbrot sets. We look
at the map
```math
f(z) = (A z)^2 + c,
```
where $A \in \mathbb{C}^{n \times n}$ and $z \in \mathbb{C}^n$ with just
$c \in \mathbb{C}$. This gives some interesting results that are not directly
analogous to the standard scalar case:

* The escape radius is no longer just $2$.
* The periodicity of the various points is weirder.
* There are (possibly) multiple attractive or repelling fixed points, not just
  $z = 0 + 0\imath$.

Additional math needed! Most of these ideas have no proofs at the moment, but
seem fun to investigate!

Install
-------

This is a Rust app and uses all the standard build infrastructure. To build it,
just run
```bash
cargo build --release
```

Usage
-----

This is currently **very experimental** and just meant for playing around. Even
so, it's nicely parallelized with `rayon` and colored. The executable takes in
JSON files that contain the matrix, the bounds, and a desired escape radius.
There are a few examples in `data/` and they look like this
```json
{
  "mat": [
    [[1.0, 0.0], [0.8, 0.0], [1.0, 0.0], [-0.5,0.0]],
    2,
    2
  ],
  "escape_radius": 3.4742662001265163,
  "upper_left": [-0.9, 0.6],
  "lower_right": [0.4, -0.6]
}
```

The matrix is given as `[[ list of entries ], nx, ny]`, where each entry is
a `[z.real, z.imag]` tuple. The entries are listed column by column. The upper
and lower corners of the rendering box are also given as `[x, y]` coordinates.
Using such a file, you can just run
```bash
netbrot --color orbit data/netbrot-2x2.json
```
to get nicely colored orbits or
```bash
netbrot --color period data/netbrot-2x2.json
```
to get nicely colored periods.

There a little script `scripts/generate-exhibits.py` that can be used to generate
some more random matrices of various sizes, but you're encouraged to just make your
own. This script can be called as
```bash
python scripts/generate-exhibits.py random --size 5 --count 10 feedforward
```

Example
-------

As a simple example, we take the matrix (see `data/readme.json`)
```math
A =
\begin{bmatrix}
1 & 0.8 \\
1 & -0.5
\end{bmatrix}

\implies

\begin{bmatrix}
z_1^{n + 1} \\
z_2^{n + 1}
\end{bmatrix}
=
\begin{bmatrix}
(z_1^n + 0.8 z_2^n)^2 + c \\
(z_1^n - 0.5 z_2^n)^2 + c
\end{bmatrix}.
```

<p align="center">
    <img src="https://github.com/alexfikl/netbrot/blob/main/docs/netbrot-2x2.png?raw=true" alt="Netbrot 2x2"/>
</p>

Obtained by running
```bash
netbrot -r 2000 -m 128 data/readme.json
```

License
-------

The code is MIT licensed (see `LICENSES/MIT.txt`). It was originally copied
from the Rust Programming example [here](https://github.com/ProgrammingRust/mandelbrot)
and has since evolved a bit.
