# SpiegeLib

Synthesizer Programming with Intelligent Exploration, Generation, and Evaluation Library.

[![PyPI version](https://badge.fury.io/py/spiegelib.svg)](https://badge.fury.io/py/spiegelib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

An object oriented Python library for research and development related to Automatic Sound Synthesizer Programming. *SpiegeLib* contains a set of classes and base classes for developing and evaluting algorithms for generating parameters and patch settings for synthesizers.

SpiegeLib is named after composer Laurie Spiegel, an early pioneer in electronic music. She is known for her work with synthesizers and for automating various aspects of the music composition process. Her philosophy for using technology and automation in music serves as motivation for this project:

*I automate whatever can be automated to be freer to focus on those aspects of music that can’t be automated. The challenge is to figure out which is which.*

## Documentation & Examples

Detailed installation instructions, API reference, and examples are available [here](https://spiegelib.github.io/spiegelib/).

An in-depth example using *SpiegeLib* for synthesizer sound matching of *Dexed*, an emulation of the Yamaha DX7, is available [here](https://spiegelib.github.io/spiegelib/examples/fm_sound_match.html)

## Features

- Classes for creating research datasets and running sound matching experiments
- Programmatic control and rendering of audio from VST synthesizers
- Audio feature extraction
- Deep learning algorithms
- Evolutionary algorithms 
- Objective & subjective evaluation tools

###### Programmatic Control of Synthesizers

Program and render audio from VST synthesizers or write your own custom synthesizer classes. **SynthVST** class provides control of VST synthesizers using the [RenderMan](https://github.com/fedden/RenderMan) library.

###### Deep Learning

Deep learning algorithms implemented using [Keras & TensorFlow](https://www.tensorflow.org/). *SpiegeLib* includes the following models which have been used in preivous work in the field of automatic synthesizer programming:

- Multi-layer Perceptron (MLP)
- Long Short-Term Memory (LSTM)
- Bi-directional Long Short Term Memory with Highway Layers (LSTM++)
- Convolutional Neural Network (CNN)

###### Evolutionary Algorithms

Evolutionary algorithms, including genetic algorithms, supported using the [DEAP framework](https://github.com/DEAP/deap). *SpiegeLib* includes the following algorithms which have been used in previous automatic synthesizer programming research:

- A basic single objective genetic algorithm (GA)
- A multi-objective non-dominated sorting genetic algorithm (NSGA III)

###### Evaluation

Tools for running both objective and subjective evaluation of experimental results are provided. Results can be evaluated objectively using the **MFCCEval** class which calcuates error and distance metrics on a set of audio file targets and estimations.

Basic subjective evaluation of results is provided in the **Subjective** class which creates a basic MUSHRA style listening test using [BeaqleJS](https://github.com/HSU-ANT/beaqlejs) and serves it to localhost so it can be taken in a browser.

