.PHONY: clean clean-test clean-pyc clean-build clean-venvs line pep8 docs dist install develop help
.DEFAULT_GOAL := help
CONDA_PACKAGES = qutip
TESTENV =
#TESTENV = MATPLOTLIBRC=tests
TESTOPTIONS = --doctest-modules --cov=qnet
TESTS = src tests docs/*.rst


define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test clean-venvs ## remove all build, test, coverage, and Python artifacts, as well as environments

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find tests src -name '*.egg-info' -exec rm -fr {} +
	find tests src -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find tests src -name '*.pyc' -exec rm -f {} +
	find tests src -name '*.pyo' -exec rm -f {} +
	find tests src -name '*~' -exec rm -f {} +
	find tests src -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/

clean-venvs: ## remove testing/build environments
	rm -fr .tox
	rm -fr .venv

lint: ## check style with flake8
	flake8 src tests

pep8: ## check style with pep8
	pep8 src tests


test:  test35 test36 ## run tests on every Python version


.venv/py35/bin/pytest:
	@conda create -y -m --override-channels -c defaults -p .venv/py35 python=3.5
	@# if the conda installation does not work, simply comment out the following line, and let pip handle it
	@conda install -y --override-channels -c defaults -c conda-forge -p .venv/py35 $(CONDA_PACKAGES)
	@.venv/py35/bin/pip install -e .[simulation,visualization,dev]

test35: .venv/py35/bin/pytest ## run tests for Python 3.5
	$(TESTENV) $< -v $(TESTOPTIONS) $(TESTS)



.venv/py36/bin/pytest:
	@conda create -y -m --override-channels -c defaults -p .venv/py36 python=3.6
	@# if the conda installation does not work, simply comment out the following line, and let pip handle it
	@conda install -y --override-channels -c defaults -c conda-forge -p .venv/py36 $(CONDA_PACKAGES)
	@.venv/py36/bin/pip install -e .[simulation,visualization,dev]

test36: .venv/py36/bin/pytest ## run tests for Python 3.6
	$(TESTENV) $< -v $(TESTOPTIONS) $(TESTS)

.venv/py36/bin/sphinx-build: .venv/py36/bin/pytest

docs: .venv/py36/bin/sphinx-build ## generate Sphinx HTML documentation, including API docs
	$(MAKE) -C docs SPHINXBUILD=../.venv/py36/bin/sphinx-build clean
	$(MAKE) -C docs SPHINXBUILD=../.venv/py36/bin/sphinx-build html
	@echo "open docs/_build/html/index.html"

coverage: test36  ## generate coverage report in ./htmlcov
	.venv/py36/bin/coverage html
	@echo "open htmlcov/index.html"

test-release: clean-build clean-pyc dist ## package and upload a release to test.pypi.org
	twine check dist/*
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: clean-build clean-pyc dist ## package and upload a release
	twine check dist/*
	twine upload dist/*


dist: clean-build clean-pyc ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	twine check dist/*
	ls -l dist

install: clean-build clean-pyc ## install the package to the active Python's site-packages
	pip install .

uninstall:  ## uinstall the package from the active Python's site-packages
	pip uninstall qnet

develop: clean-build clean-pyc ## install the package to the active Python's site-packages, in develop mode
	pip install -e .

develop-test: develop ## run tests within the active Python environment
	$(TESTENV) pytest -v $(TESTOPTIONS) $(TESTS)

develop-docs: develop  ## generate Sphinx HTML documentation, including API docs, within the active Python environment
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	@echo "open docs/_build/html/index.html"
