#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Init for synth components
"""

from .synth_base import SynthBase

try:
    import librenderman
    from .synth_vst import SynthVST
except:
    print("librenderman package not installed, SynthVST class is unavailable. To use VSTs please install librenderman.")
    print("https://spiegelib.github.io/spiegelib/getting_started/installation.html")
