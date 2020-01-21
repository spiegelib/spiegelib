Contribution Guide
==================

General
-------

Contributions to *spiegel* are welcome and encouraged. See the following
instructions to get setup for development.

Installation
------------

It is recommended to develop inside of an `anaconda <https://www.continuum.io/>`_
virtual environment.

1) Clone the repo::

	$ git clone https://github.com/jorshi/spiegel

2) Create a new ``conda`` from environment file::

	$ cd spiegel
	$ conda env create -f spiegel-env.yml

	. . .

	#
	# To activate this environment, use
	#
	#     $ conda activate spiegel-env
	#
	# To deactivate an active environment, use
	#
	#     $ conda deactivate

3) Activate the ``conda`` env::

	$ conda activate spiegel-env

4) Add *spiegel* in development mode to conda environment::

	$ pip install -e .

5) Install librenderman (Mac OSX only currently)

	`librenderman <https://github.com/fedden/RenderMan>`_ is a python library for programmatically interacting with VST synthesizers.
	It uses `boost-python <https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/index.html>`_ to wrap
	a `JUCE <https://juce.com/>`_ application. Getting it running inside of a ``conda`` env is not the most
	straightforward thing and hopefully we can work to get it packaged in conda at some point.

	a) `Download and install JUCE <https://shop.juce.com/get-juce>`_


	b) Get the boost-python headers, recommended to use Brew::

		$ brew install boost-python3


	c) If you haven't already, move out of the *spiegel* project directory to somewhere you would like to download librenderman


	d) Clone the librenderman repo::

		$ git clone https://github.com/fedden/RenderMan.git


	e) Open the RenderMan jucer file in ProJucer::

		$ cd RenderMan
		$ open RenderMan.jucer

	f) Update Xcode exporter

		For the Xcode (MacOSX) exporter modify the extra linker flags. Remove ``-lpython2.7``
		Change ``-lboost-python`` to ``-lboost-python37`` and add ``-undefined dynamic_lookup``

		.. image:: images/linker_flags.png
