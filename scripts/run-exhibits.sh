#!/usr/bin/bash

# SPDX-FileCopyrightText: 2024 Alexandru Fikl <alexfikl@gmail.com>
# SPDX-License-Identifier: MIT

set -Eeuo pipefail

function with_echo() {
    echo "+++" "$@"
    nice "$@"
}

suffix=$(date "+%Y%m%d")
resolution='128'

for filename in $@; do
    with_echo ./target/release/netbrot \
        --render mandelbrot \
        --resolution ${resolution} \
        --maxit 128 \
        --outfile "${filename%.json}-x${resolution}-${suffix}.png" \
        "${filename}"
done
