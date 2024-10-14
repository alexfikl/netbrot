PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

# {{{ formatting

format: rustfmt rufffmt	shfmt		## Run all formatting scripts
.PHONY: format

rustfmt:						## Run rustfmt
	cargo fmt -- src/*.rs
	@echo -e "\e[1;32mrustfmt clean!\e[0m"
.PHONY: rustfmt

rufffmt:						## Run ruff format
	ruff format scripts experiments
	ruff check --fix --select=I scripts experiments
	ruff check --fix --select=RUF022 scripts experiments
	@echo -e "\e[1;32mruff format clean!\e[0m"

shfmt:							## Run shfmt format
	shfmt --write --language-dialect bash --indent 4 scripts/*.sh
	@echo -e "\e[1;32mshfmt clean!\e[0m"

# }}}

# {{{ linting

lint: typos reuse ruff clippy	## Run linting checks
.PHONY: lint

typos:			## Run typos over the source code and documentation
	typos --sort
	@echo -e "\e[1;32mtypos clean!\e[0m"
.PHONY: typos

reuse:			## Check REUSE license compliance
	$(PYTHON) -m reuse lint
	@echo -e "\e[1;32mREUSE compliant!\e[0m"
.PHONY: reuse

ruff:			## Run ruff checks over the source code
	ruff check scripts experiments
	@echo -e "\e[1;32mruff clean!\e[0m"
.PHONY: ruff

clippy:			## Run clippy lint checks
	cargo clippy --all-targets --all-features
	@echo -e "\e[1;32mclippy clean!\e[0m"

# }}}

# {{{ building

test:			## Run tests
	cargo test --all-features
.PHONY: test

build:			## Build the project in debug mode
	cargo build --locked --all-features --verbose
.PHONY: build

release:		## Build project in release mode
	cargo build --locked --all-features --release
.PHONY: release

windows:		## Cross compile for windows
	cargo build --target x86_64-pc-windows-gnu --locked --all-features --release
.PHONY: windows

# }}}

# {{{ gallery

mat-default:			## Generate default test matrices
	$(PYTHON) scripts/generate-exhibits.py \
		--overwrite \
		--outfile data/exhibit-example.json \
		random --type fixed
.PHONY: mat-default

mat-structural:			## Convert example structural data to npz
	$(PYTHON) scripts/generate-exhibits.py \
		--overwrite \
		--outfile data/structural.json \
		--xlim '-3.75' '1.25' \
		--ylim '-2.5' '2.5' \
		convert \
		-n 'Structural_Conn' \
		data/structural_scaled_by_task_max.mat
.PHONY: mat-structural

# }}}
