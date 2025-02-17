PYTHON := "python -X dev"

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc("Run all formatting scripts over the source code")]
format: justfmt isort black rustfmt shfmt

[doc("Run just --fmt over the justfile")]
justfmt:
    just --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

[doc("Run ruff isort fixes over the source code")]
isort:
    ruff check --fix --select=I scripts
    ruff check --fix --select=RUF022 scripts
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc("Run ruff format over the source code")]
black:
    ruff format scripts
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc("Run rustfmt over the source code")]
rustfmt:
    cargo fmt -- src/*.rs
    @echo -e "\e[1;32mrustfmt clean!\e[0m"

[doc("Run shfmt over the source code")]
shfmt:
    shfmt --write --language-dialect bash --indent 4 scripts/*.sh
    @echo -e "\e[1;32mshfmt clean!\e[0m"

# }}}
# {{{ linting

[doc("Run all linting checks over the source code")]
lint: typos reuse ruff clippy

[doc("Run typos over the source code and documentation")]
typos:
    typos --sort --format=brief scripts
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc("Check REUSE license compliance")]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc("Run ruff checks over the source code")]
ruff format="full":
    ruff check --output-format={{ format }} scripts
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc("Run clippy lint checks")]
clippy:
    cargo clippy --all-targets --all-features
    @echo -e "\e[1;32mclippy clean!\e[0m"

# }}}
# {{{ building

[doc("Run rust tests")]
test $RUST_BACKTRACE="1":
    cargo test --all-features

[doc("Build project in debug mode")]
debug:
    cargo build --locked --all-features --verbose

[doc("Build project in release mode")]
release:
    cargo build --locked --all-features --release

# }}}
# {{{

[doc("Run the main executable with the given arguments")]
run *args: release
    nice ./target/release/netbrot {{ args }}

attr filename period="1":
    @just run --render attractive --color period-matlab -r 1200 \
        --period {{ period }} \
        --outfile {{ filename }}-period{{ period }}.png \
        {{ filename }}.json

[doc("Remove various generated files")]
clean:
    rm -rf data/*.png
    rm -rf *.mat
    rm -rf *.npz
    rm -rf *.zip

[doc("Remove all generated files and caches")]
purge: clean
    rm -rf target
    rm -rf .ruff_cache

[doc("Generate README image")]
readme-image:
    @just run --resolution 2048 --maxit 128 data/readme.json -o docs/netbrot-2x2.png

# }}}
