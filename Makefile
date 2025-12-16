.PHONY: build install devinstall check format test clean

build: clean
	python3 -m build

install: build
	python3 -m install .

devinstall: build
	python3 -m install -e .[dev]

check:
	python3 -m mypy descreen

format:
	python3 -m black -l 300 descreen

test:
	python3 -X dev -m descreen data/mock/wh.png --model basic

clean:
	python3 -c 'import shutil; shutil.rmtree("dist", ignore_errors=True)'
	python3 -c 'import shutil; shutil.rmtree("build", ignore_errors=True)'
	python3 -c 'import shutil; shutil.rmtree("deep_descreen.egg-info", ignore_errors=True)'
	python3 -c 'import shutil; shutil.rmtree(".mypy_cache", ignore_errors=True)'
