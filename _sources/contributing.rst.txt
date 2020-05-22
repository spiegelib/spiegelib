Contribution Guide
==================

General
-------

Contributions to *SpiegeLib* are welcome and encouraged. See the following
instructions to get setup for development.

Requirements
------------

.. code-block::

	# Python Version
	python >= 3.6

	# Python Packages
	numpy >= 1.18.0
	matplotlib >= 3.2.0
	librosa >= 0.7.2
	tensorflow >= 2.1.0
	deap >= 1.3.1
	tqdm >= 4.43.0
	librenderman

	# Dev Requirements
	sphinx
	sphinx_rtd_theme
	ipython
	pytest
	pytest-datadir
	pytest-mpl
	tox
	twine


Installation
------------

1) Anaconda development environment

   We recommend developing from an Anaconda environment. Create a new conda environment::

		$ conda create --name spiegelib_dev python=3.7 numpy matplotlib
  		...
		$ conda activate spiegelib_dev

2) Fork the `spiegelib repo <https://github.com/spiegelib/spiegelib>`_ and clone your forked repository

3) Install SpiegeLib in development mode with additional dev requirements to ``conda`` environment::

	$ cd spiegelib
	$ pip install -e .[dev]

	If you're using zsh on Mac then you may need to use this command:

	$ pip install -e ".[dev]"

4) Install `RenderMan <https://github.com/fedden/RenderMan>`_ in conda environment. See instructions :ref:`here<librenderman_conda_install>`

5) Create a new branch and develop! Submit a pull request when you are ready to add changes.


Tests
-----

Unit tests are written using `pytest <https://docs.pytest.org/en/latest/>`_ and
can be run with the following command when in the project root::

	$ python -m pytest

Testing coverage is available with `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/index.html>`_::

	# To run with coverage reporting
	$ python -m pytest --cov=./src

	# And coverage report with missing lines
	$ python -m pytest --cov=./src --cov-report=term-missing


Documentation
-------------

This documentation is all written in restructured text (RST) with `sphinx-doc <https://www.sphinx-doc.org/en/master/usage/configuration.html>`_.
Documentation is all written in the gh-pages branch in docs directory::

	spiegelib/docs/source

To build documentation::

	# First make sure you are in the root docs directory
	$ cd docs

	# Then run
	$ make html

	# To copy the built documentation to the project root
	$ ./copy.sh

	# Documentation can be viewed locally
	$ open ../index.html

When developing, please add class documentation using `Google docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

To update and write documentation make sure the gh-pages branch is up-to-date with your
development branch as documentation from comments in python files in the source code
is used to automatically generate reference documentation. Make changes and additions
to the documentation source rst files. Then build using the process listed above.
