Contribution Guide
==================

General
-------

Contributions to *spiegel* are welcome and encouraged. See the following
instructions to get setup for development.

Installation
------------

It is recommended to develop inside of an `anaconda <https://www.continuum.io/>`_
virtual environment. These instructions are for setting up for development within a
``conda`` env, see `installation instructions <https://www.anaconda.com/distribution/#download-section>`_ for anaconda,
download and install the python 3 version.

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

4) Add *spiegel* in development mode to ``conda`` environment::

	$ pip install -e .

5) Install librenderman (Mac OSX only currently)

	`librenderman <https://github.com/fedden/RenderMan>`_ is a python library for programmatically interacting with VST synthesizers.
	It uses `boost-python <https://www.boost.org/doc/libs/1_66_0/libs/python/doc/html/index.html>`_ to wrap
	a `JUCE <https://juce.com/>`_ application. Getting it running inside of a ``conda`` env is not the most
	straightforward thing and hopefully we can work to get it packaged in anaconda at some point. But, for now:

	a) `Download and install JUCE <https://shop.juce.com/get-juce>`_

	b) If you haven't already, move out of the *spiegel* project directory to somewhere you would like to download librenderman

	c) Clone the librenderman repo::

		$ git clone https://github.com/fedden/RenderMan.git

	e) Open the RenderMan python3 jucer file in ProJucer::

		$ cd RenderMan
		$ open RenderMan-py36.jucer

	f) Update Xcode exporter

		For the Xcode (MacOSX) exporter modify the *Extra Linker Flags*: remove ``-lpython3.6m``
		Change ``-lboost-python`` to ``-lboost-python37`` and add ``-undefined dynamic_lookup``

		.. image:: images/linker_flags.png

	g) Update Xcode Debug export

		| A sub category of the Xcode (MacOSX) exporter is the Debug specific options. Here we want
			to change the header and library search paths to look for headers and libs in our ``conda`` env.
			The exact location will depend on where your ``conda`` environments are on your system, which was
			determined when you installed anaconda. The default location on MacOSX is ``/Users/<your-username>/anaconda3``.
		|
		| The following instructions will refer to that location, whatever it is on your system, as ``<path-to-anaconda3>``.
		|
		| For the *Header Search Paths*, remove the existing paths and add ``<path-to-anaconda3>/envs/spiegel-env/include`` and
			``<path-to-anaconda3>/envs/spiegel-env/include/python3.7m``.
		|
		| Using the default anaconda path:

		.. image:: images/header_search_paths.png

		|
		| For the *Extra Library Search Paths*, remove the existing paths and add ``<path-to-anaconda3>/envs/spiegel-env/lib``
		|
		| Using the default anaconda path:

		.. image:: images/extra_lib_path.png

	h) Open Xcode and build

		Open Xcode from the Projucer by clicking on the Xcode icon

		.. image:: images/Xcode_projucer.png

		|

		Build the library in Xcode

		.. image:: images/build_xcode.png

		|

		Celebrate successful build!

	i) Rename the built library and move to conda env::

		$ cd <path-to-RenderMan>/Builds/MacOSX/build/Debug
		$ mv librenderman.so.dylib librenderman.so
		$ mv librenderman.so <path-to-anaconda3>/envs/spiegel-env/lib/python3.7/site-packages/

	j) Test librenderman. Make sure the ``conda`` *spiegel-env* is activated before running python::

		$ python
		>>> import librenderman as rm
		JUCE v<Juce version>
		>>> engine = rm.RenderEngine(44100, 512, 512)
		>>>

If you made it this far without errors, then you should be good to go!
