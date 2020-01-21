#!/usr/bin/env python


from __future__ import print_function
import os
import sys
import argparse

from spiegel.synth.synth_vst import SynthVST
from spiegel.features.mfcc import MFCC
from spiegel import DatasetGenerator

def loadPatch():

    synth.randomizePatch()
    synth.renderPatch()

    audio = synth.getAudio()

    mfcc = MFCC()
    mfccs = mfcc.getFeatures(audio)

    scipy.io.wavfile.write('./test.wav', 44100, audio)


def generateDataset(synth):

    features = MFCC()
    #features.loadNormalizers("./data/normalizers.pkl")

    generator = DatasetGenerator(synth, features=features, outputFolder="./data", saveAudio=True)
    generator.generate(1000, filePrefix="train_", fitNormalizers=True)
    generator.generate(100, filePrefix="test_")

    generator.saveNormalizers('normalizers.pkl')


def main(arguments):

    synth = SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst", noteLengthSecs=0.5, renderLengthSecs=0.75)
    parameters = synth.getParameters()
    patch = []
    overridden = []

    algorithm_number = 18
    alg = (1.0 / 32.0) * float(algorithm_number - 1) + 0.001

    overriden_parameters = [(0, 1.0), (1, 0.0), (2, 1.0), (3, 0.0), (4, alg)]

    other_params = [((i + 5), 0.5) for i in range(18)]

    operator_one = [((i + 23), 0.0) for i in range(22)]
    operator_two = [((i + 45), 0.0) for i in range(22)]
    operator_thr = [((i + 67), 0.0) for i in range(22)]
    operator_fou = [((i + 89), 0.0) for i in range(22)]
    operator_fiv = [((i + 111), 0.0) for i in range(22)]
    operator_six = [((i + 133), 0.0) for i in range(22)]

    # overriden_parameters.extend(operator_one)
    overriden_parameters.extend(operator_two)
    overriden_parameters.extend(operator_thr)
    overriden_parameters.extend(operator_fou)
    overriden_parameters.extend(operator_fiv)
    overriden_parameters.extend(operator_six)
    overriden_parameters.extend(other_params)


    synth.setOverriddenParameters(overriden_parameters)

    generateDataset(synth)





if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
