#!/usr/bin/bash

# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

function with_echo() {
    echo "+++" "$@"
    nice "$@"
}

suffix=$(date "+%Y%m%d")

for filename in $@; do
    with_echo ./target/release/netbrot \
        --render mandelbrot \
        --resolution 128 \
        --maxit 128 \
        --outfile "${filename%.json}-1200x1200-${suffix}.png" \
        "${filename}"
done
