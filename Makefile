PROJECT_NAME = QNET
PACKAGES =  pip numpy matplotlib scipy sympy ipython bokeh pytest sphinx nose ply cython coverage
TESTPYPI = https://testpypi.python.org/pypi

#TESTOPTIONS = --doctest-modules --cov=qnet
TESTOPTIONS = --doctest-modules --cov=qnet
TESTS = qnet tests
# You may redefine TESTS to run a specific test. E.g.
#     make test TESTS="tests/algebra"

VERSION = $(shell grep __version__ < qnet/__init__.py | sed 's/.*"\(.*\)"/\1/')

develop:
	pip install --process-dependency-links -e .[simulation,circuit_visualization,dev]

install:
	pip install --process-dependency-links .

uninstall:
	pip uninstall $(PROJECT_NAME)

sdist:
	python setup.py sdist

upload:
	python setup.py register
	python setup.py sdist upload

test-upload:
	python setup.py register -r $(TESTPYPI)
	python setup.py sdist upload -r $(TESTPYPI)

test-install:
	pip install -i $(TESTPYPI) $(PROJECT_NAME)

clean:
	@rm -rf build
	@rm -rf dist
	@rm -rf QNET.egg-info
	@find . -iname *pyc | xargs rm -f
	@find . -iname __pycache__ | xargs rm -rf
	@make -C docs clean
	@rm -rf htmlcov
	@rm -rf .cache
	@rm -f .coverage

distclean: clean
	@rm -rf .venv

.venv/py35/bin/py.test:
	@conda create -y -m -p .venv/py35 python=3.5 $(PACKAGES)
	@find .venv/py35 -iname matplotlibrc | xargs sed -i.bak 's/backend\s*:\s*Qt5Agg/backend: Agg/'
	@# if the conda installation does not work, simply comment out the following line, and let pip handle it
	@conda install -y -c conda-forge -p .venv/py35 qutip
	@.venv/py35/bin/pip install --process-dependency-links -e .[simulation,circuit_visualization,dev]

test35: .venv/py35/bin/py.test
	$< -v $(TESTOPTIONS) $(TESTS)

test: test35

coverage: test35
	@rm -rf htmlcov/index.html
	.venv/py35/bin/coverage html

doc: .venv/py35/bin/py.test
	@rm -f docs/API/*.rst
	$(MAKE) -C docs SPHINXBUILD=../.venv/py35/bin/sphinx-build SPHINXAPIDOC=../.venv/py35/bin/sphinx-apidoc html
	@echo "Documentation is in docs/_build/html"

.PHONY: install develop uninstall upload test-upload test-install sdist clean \
test test35 doc coverage distclean
