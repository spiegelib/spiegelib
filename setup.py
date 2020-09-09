#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__version__ = '0.0.4'
__author__ = "Jordie Shier"
__contact__ = "jordieshier@gmail.com"
__url__ = "https://github.com/spiegelib/spiegelib"
__license__ = "MIT"


with open("README.md", "r", encoding='utf-8') as f:
    readme = f.read()

setup(
    name='spiegelib',
    version=__version__,
    author=__author__,
    author_email=__contact__,
    description='Synthesizer Programming with Intelligent Exploration, Generation, and Evaluation Library',
    long_description=readme,
    long_description_content_type='text/markdown',
    url=__url__,
    licence=__license__,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        '': ['beaqlejs/*', 'beaqlejs/img/*'],
    },
    scripts=[
        'scripts/soundmatch.httpserver',
        'scripts/soundmatch.oscserver',
        'scripts/synthml.server'
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'librosa',
        'tqdm',
        'matplotlib',
        'tensorflow',
        'deap',
        "importlib_resources ; python_version<'3.7'",
        'numba==0.48',
        'scipy==1.4.1'
    ],
    extras_require={
        'dev': [
            'sphinx',
            'sphinx_rtd_theme',
            'ipython',
            'pytest',
            'pytest-datadir',
            'pytest-mpl',
            'tox',
            'twine'
        ],
    }
)
