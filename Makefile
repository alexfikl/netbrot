PYTHON?=python -X dev

all: help

help: 			## Show this help
	@echo -e "Specify a command. The choices are:\n"
	@grep -E '^[0-9a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[0;36m%-12s\033[m %s\n", $$1, $$2}'
	@echo ""
.PHONY: help

format: rustfmt rufffmt			## Run all formatting scripts
.PHONY: format

rustfmt:						## Run rustfmt
	cargo fmt -- src/*.rs
	@echo -e "\e[1;32mrustfmt clean!\e[0m"
.PHONY: rustfmt

rufffmt:						## Run ruff format
	ruff format scripts
	ruff check --fix --select=I scripts
	ruff check --fix --select=RUF022 scripts
	@echo -e "\e[1;32mruff format clean!\e[0m"

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
	ruff check scripts
	@echo -e "\e[1;32mruff clean!\e[0m"
.PHONY: ruff

clippy:			## Run clippy lint checks
	cargo clippy
	@echo -e "\e[1;32mclippy clean!\e[0m"

build:			## Build the project in debug mode
	cargo build --verbose
.PHONY: build

release:		## Build project in release mode
	cargo build --release
.PHONY: release
