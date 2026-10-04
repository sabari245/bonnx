# bonnx — web app (bun + vite) and the Python launcher.   `make help` lists targets.

BUN    ?= bun
PY     ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
PYPATH  = PYTHONPATH=python
MIRROR ?= https://registry.npmmirror.com
MODEL  ?=
OUT    ?= $(if $(MODEL),$(basename $(notdir $(MODEL))).html,)
PORT   ?= 8080

SRC := $(shell find src index.html -type f 2>/dev/null)

.PHONY: help install install-mirror dev build single test test-js test-py package viz serve clean

help:
	@echo "  make install         bun install"
	@echo "  make install-mirror  bun install via a mirror + fix bun.lock (when bun hangs on IPv6-only DNS, see README)"
	@echo "  make dev             vite dev server with hot reload"
	@echo "  make build           production build -> dist/"
	@echo "  make single          single-file build -> dist-single/index.html"
	@echo "  make test            JS tests (vitest) and Python tests (unittest)"
	@echo "  make package         copy the builds into python/bonnx/ so a pip install is self-contained"
	@echo "  make serve MODEL=m.onnx [PORT=8080]   serve a model on localhost (builds the app first if needed)"
	@echo "  make viz   MODEL=m.onnx [OUT=m.html]  one self-contained HTML file (replaces the old scripts/model_viz.py)"

install:
	$(BUN) install

# bun resolves registry.npmjs.org to IPv6 first and does not fall back to IPv4, so on hosts with broken IPv6 it
# hangs forever. Resolve through a mirror, then point the lockfile back at the canonical registry.
install-mirror:
	$(BUN) install --registry=$(MIRROR)
	sed -i 's#$(MIRROR)/#https://registry.npmjs.org/#g' bun.lock

dev:
	$(BUN) run dev

dist/index.html: $(SRC) package.json vite.config.ts tsconfig.json
	$(BUN) run build

dist-single/index.html: $(SRC) package.json vite.config.ts tsconfig.json
	$(BUN) run build:single

build: dist/index.html
single: dist-single/index.html

test: test-js test-py

test-js:
	$(BUN) run test

test-py:
	$(PYPATH) $(PY) -m unittest discover -s python/tests -t python/tests

package: dist/index.html dist-single/index.html
	rm -rf python/bonnx/web python/bonnx/web_single
	mkdir -p python/bonnx/web_single
	cp -r dist python/bonnx/web
	cp dist-single/index.html python/bonnx/web_single/index.html
	@echo "bundled web app into python/bonnx/ (pip install ./python now ships it)"

serve: dist/index.html
	@test -n "$(MODEL)" || { echo "usage: make serve MODEL=path/to/model.onnx"; exit 2; }
	$(PYPATH) $(PY) -m bonnx "$(MODEL)" --port $(PORT)

viz: dist-single/index.html
	@test -n "$(MODEL)" || { echo "usage: make viz MODEL=path/to/model.onnx [OUT=out.html]"; exit 2; }
	$(PYPATH) $(PY) -m bonnx "$(MODEL)" --out "$(OUT)"

clean:
	rm -rf dist dist-single python/bonnx/web python/bonnx/web_single
