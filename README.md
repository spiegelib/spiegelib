# SpiegeLib

Synthesizer Programming with Intelligent Exploration, Generation, and Evaluation Library.

[![PyPI version](https://badge.fury.io/py/spiegelib.svg)](https://badge.fury.io/py/spiegelib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

An object oriented Python library for research and development related to Automatic Synthesizer Programming. *SpiegeLib* contains a set of classes and base classes for developing and evaluating algorithms for generating parameters and patch settings for synthesizers.

SpiegeLib is named after composer Laurie Spiegel, an early pioneer in electronic music. She is known for her work with synthesizers and for automating various aspects of the music composition process. Her philosophy for using technology and automation in music serves as motivation for this project:

*I automate whatever can be automated to be freer to focus on those aspects of music that can’t be automated. The challenge is to figure out which is which.*

## Documentation & Examples

Detailed installation instructions, API reference, and examples are available [here](https://spiegelib.github.io/spiegelib/).

An example using *SpiegeLib* for synthesizer sound matching of *Dexed*, an emulation of the Yamaha DX7, is available [here](https://spiegelib.github.io/spiegelib/examples/fm_sound_match.html)

## Features

- Classes for creating research datasets and running sound matching experiments
- Programmatic control and rendering of audio from VST synthesizers
- Audio feature extraction
- Deep learning algorithms
- Evolutionary algorithms 
- Objective & subjective evaluation tools

##### Programmatic Control of Synthesizers

Program and render audio from VST synthesizers or write your own custom synthesizer classes. ``SynthVST`` class provides control of VST synthesizers using the [RenderMan](https://github.com/fedden/RenderMan) library.

##### Deep Learning

Deep learning algorithms implemented using [Keras & TensorFlow](https://www.tensorflow.org/). *SpiegeLib* includes the following models which have been used in preivous work in the field of automatic synthesizer programming:

- Multi-layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Bi-directional Long Short Term Memory with Highway Layers (LSTM++)
- Convolutional Neural Network (CNN)

The ``DatasetGenerator`` class can be used to create datasets of synthesizer patches for training and validating deep learning models.

##### Evolutionary Algorithms

Evolutionary algorithms, including genetic algorithms, supported using the [DEAP framework](https://github.com/DEAP/deap). *SpiegeLib* includes the following algorithms which have been used in previous automatic synthesizer programming research:

- A basic single objective genetic algorithm (GA)
- A multi-objective non-dominated sorting genetic algorithm (NSGA III)

##### Evaluation

Tools for running both objective and subjective evaluation of experimental results are provided. Results can be evaluated objectively using the ``MFCCEval`` class which calcuates error and distance metrics on a set of audio file targets and estimations.

Basic subjective evaluation of results is provided in the ``Subjective`` class which creates a basic MUSHRA style listening test using [BeaqleJS](https://github.com/HSU-ANT/beaqlejs) and serves it to localhost so it can be taken in a browser.

## Installation

Detailed instructions on installing *SpeigeLib* in a conda environment are available [here](https://spiegelib.github.io/spiegelib/getting_started/installation.html)

*SpiegeLib* is available via pip:
`pip install spiegelib`

**Note: Requires Python > 3.6**

All requirements except for [RenderMan](https://github.com/fedden/RenderMan) will be installed via pip. RenderMan must be installed manually. We have instructions for installing RenderMan in a conda environment [here](https://spiegelib.github.io/spiegelib/getting_started/installation.html#installing-renderman-in-an-anaconda-environment).

**Currently RenderMan is only available for MacOSX and Linux operating systems.** However, there was a recent pull request made to RenderMan to add support for Windows - it has not been merged yet and we have not tested it, but you Windows users might be interested in checking that out, [here](https://github.com/DBraun/RenderMan/tree/windows).

## Issues and Contributions

Please report and issues or desires for enhancements on the GitHub issues page. Contributions are also welcome and encouraged. Check out the [contributions guide](https://spiegelib.github.io/spiegelib/contributing.html) for more information on getting setup for development.

##### Planned Future Additions

*Parameter Estimation / Generation Models*
- [ ] More robust CNNs, see [Barkan et al.](https://ieeexplore.ieee.org/abstract/document/8854832?casa_token=7_TqusED43UAAAAA:UFEPg_f7LxWyRa-oTezaA2YW8tgxQpZ9WtBArJdPRcelynwFsO4wkWTAtRs5zoZL5lLjBWIumTY)
- [ ] Generative model, see [Esling et al.](https://www.mdpi.com/2076-3417/10/1/302)
- [ ] Particle Swarm Optimization, see [Heise et al.](http://www.aes.org/e-lib/online/browse.cfm?elib=15053)
- [ ] Hill Climber optimizer (for benchmarking), see [Yee-King et al.](https://ieeexplore.ieee.org/abstract/document/8323327?casa_token=795OBbWb74QAAAAA:iXxMiGgAJ32MMWYpRlqPy0xvAxwztedvv3sZpaJZ9t-O9-nNJmtTteiEjhtB2h0CQ6S8mHhFbII)

## Acknowledgements

Thank you to Matthew Yee-King, Leon Fedden, and Mark d'Inverno for their work on [*Automatic Programming of VST Sound Synthesizers Using Deep Networks and Other Techniques*](https://ieeexplore.ieee.org/abstract/document/8323327?casa_token=795OBbWb74QAAAAA:iXxMiGgAJ32MMWYpRlqPy0xvAxwztedvv3sZpaJZ9t-O9-nNJmtTteiEjhtB2h0CQ6S8mHhFbII) and the research code they released with that project. Their work served as inspiration for *SpiegeLib* and was an invaluable resource in beginning to put together this library.

## Contributors

 - Jordie Shier https://github.com/jorshi
