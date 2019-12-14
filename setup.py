#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(
    name='spiegel',
    version='1.0.0',
    licence='LICENSE.txt',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
