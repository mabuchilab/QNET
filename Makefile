PROJECT_NAME = QNET
PACKAGES =  pip numpy matplotlib scipy sympy ipython bokeh pytest sphinx nose ply cython
TESTPYPI = https://testpypi.python.org/pypi

#TESTOPTIONS = --doctest-modules
TESTOPTIONS = -s -x --pdb
TESTS = qnet/misc/test/test_qsd_codegen.py
# You may redefine TESTS to run a specific test. E.g.
#     make test TESTS="qnet/algebra/test"

VERSION = $(shell grep __version__ < qnet/__init__.py | sed 's/.*"\(.*\)"/\1/')

DOC = qnet-doc-$(VERSION)

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
	@rm -rf QDYN.egg-info
	@find . -iname *pyc | xargs rm -f
	@find . -iname __pycache__ | xargs rm -rf
	@rm -rf $(DOC) $(DOC).tgz

.venv/py27/bin/py.test:
	@conda create -y -m -p .venv/py27 python=2.7 $(PACKAGES)
	@.venv/py27/bin/pip install --no-use-wheel qutip
	@.venv/py27/bin/pip install --process-dependency-links -e .[simulation,circuit_visualization,dev]

.venv/py33/bin/py.test:
	@conda create -y -m -p .venv/py33 python=3.3 $(PACKAGES)
	@.venv/py33/bin/pip install --no-use-wheel qutip
	@.venv/py33/bin/pip install --process-dependency-links -e .[simulation,circuit_visualization,dev]

.venv/py34/bin/py.test:
	@conda create -y -m -p .venv/py34 python=3.4 $(PACKAGES)
	@.venv/py34/bin/pip install --no-use-wheel qutip
	@.venv/py34/bin/pip install --process-dependency-links -e .[simulation,circuit_visualization,dev]

test27: .venv/py27/bin/py.test
	$< -v $(TESTOPTIONS) $(TESTS)

test33: .venv/py33/bin/py.test
	$< -v $(TESTOPTIONS) $(TESTS)

test34: .venv/py34/bin/py.test
	$< -v $(TESTOPTIONS) $(TESTS)

test: test27 test33 test34

doc:
	make -C docs html
	@rm -rf $(DOC)
	@cp -r docs/_build/html $(DOC)
	tar -c $(DOC) | gzip > $(DOC).tgz
	@rm -rf $(DOC)

.PHONY: install develop uninstall upload test-upload test-install sdist clean \
test test27 test33 test34 doc
