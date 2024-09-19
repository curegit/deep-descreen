.PHONY: build install devinstall preview publish check format test testtrain clean

build: clean
	python3 -m build

install: build
	pip3 install .

devinstall: build
	pip3 install -e .[dev]

preview: build
	false && python3 -m twine upload -u __token__ --repository-url "https://test.pypi.org/legacy/" dist/*

publish: build
	false && python3 -m twine upload -u __token__ --repository-url "https://upload.pypi.org/legacy/" dist/*

check:
	python3 -m mypy descreen

format:
	python3 -m black -l 300 descreen

test:
	python3 -X dev -m descreen data/mock/wh.png --model basic
	python3 -X dev -m descreen data/mock/wh.png --model unet

testtrain:
	python3 -X dev -m descreen.training data/mock data/mock data/mock -e 3 -m basic -p " " -d testrun
	python3 -X dev -m descreen data/mock/wh.png --ddbin testrun/basic-final.ddbin

clean:
	python3 -c 'import shutil; shutil.rmtree("dist", ignore_errors=True)'
	python3 -c 'import shutil; shutil.rmtree("build", ignore_errors=True)'
	python3 -c 'import shutil; shutil.rmtree("deep_descreen.egg-info", ignore_errors=True)'
	python3 -c 'import shutil; shutil.rmtree(".mypy_cache", ignore_errors=True)'
	python3 -c 'import shutil; shutil.rmtree("testrun", ignore_errors=True)'
