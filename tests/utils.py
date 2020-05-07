"""
Utility functions for testing
"""

import numpy as np


def make_test_sine(size, hz, rate=44100):
    samples = np.zeros(size)
    phaseIncrement = (2*np.pi) / (float(rate) / float(hz))
    phase = 0.0
    for i in range(size):
        samples[i] = np.sin(phase)
        phase = phase + phaseIncrement

    return samples


def make_test_cosine(size, hz, rate=44100):
    samples = np.zeros(size)
    phaseIncrement = (2*np.pi) / (float(rate) / float(hz))
    phase = 0.0
    for i in range(size):
        samples[i] = np.cos(phase)
        phase = phase + phaseIncrement

    return samples
