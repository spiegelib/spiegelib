#!/usr/bin/env python

"""
Audio feature extraction utilities
"""

import numpy as np

spectrum_types = ['complex', 'magnitude', 'power', 'magnitude_phase', 'power_phase']
def convert_spectrum(spectrum, type, dtype=np.float32, complex_dtype=np.complex64):
    """
    Convert a raw spectrum to magnitue, power, magnitude with phase, or power
    with phase

    :param spectrum: raw audio spectrum with real/imag values to convert
    :type spectrum: np.ndarray
    :param type: type of conversion to apply ('magnitude', 'power', 'magnitude_phase',
        'power_phase')
    :type type: str
    :param dtype: number type of regular (non-complex) numbers to return, defaults to np.float32
    :type dtype: np number type, optional
    :param complex_dtype: complex number type, defaults to np.complex64
    :type complex_dtype: np complex number type, optional
    :returns: converted spectrum
    :rtype: np.ndarray
    """

    if not type in spectrum_types:
        raise ValueError("type must be one of %s, recevied %s" % (spectrum_types, type))

    # Convert to desired output format and data type
    if type == 'magnitude':
        return np.array(np.abs(spectrum), dtype=dtype)

    elif type == 'power':
        return np.array(np.abs(spectrum)**2, dtype=dtype)

    elif type == 'magnitude_phase':
        magnitude = np.array(np.abs(spectrum), dtype=dtype)
        phase = np.array(np.angle(spectrum), dtype=dtype)
        return np.stack((magnitude, phase), axis=-1)

    elif type == 'power_phase':
        power = np.array(np.abs(spectrum)**2, dtype=dtype)
        phase = np.array(np.angle(spectrum), dtype=dtype)
        return np.stack((power, phase), axis=-1)

    # If we get here, then the type is complex, so return a complex array
    # with an updated data type
    return np.array(spectrum, dtype=complex_dtype)
