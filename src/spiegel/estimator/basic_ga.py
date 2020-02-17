#!/usr/bin/python
"""
Basic single objective genetic algorithm
"""

import numpy as np
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from spiegel.evaluation.audio_eval_base import AudioEvalBase
from spiegel.estimator.estimator_base import EstimatorBase
from spiegel.synth.synth_base import SynthBase
from spiegel.features.features_base import FeaturesBase


class BasicGA(EstimatorBase):
    """
    """

    def __init__(self, synth, features, seed=None):
        """
        Constructor
        """
        super().__init__()

        if not isinstance(synth, SynthBase):
            raise TypeError("synth must be of type SynthBase")

        self.synth = synth
        self.numParams = len(synth.getPatch())

        if not isinstance(features, FeaturesBase):
            raise TypeError("features must be of type FeaturesBase")

        self.features = features
        self.targe = None

        random.seed(seed)
        self.setup()


    def setup(self):
        """
        Setup genetic algorithm
        """

        # Create individual types
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Setup toolbox
        self.toolbox = base.Toolbox()

        # Attribute generator
        self.toolbox.register("attr_float", random.random)

        # Structure initializers
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            self.numParams
        )
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)


    def fitness(self, individual):
        """
        Evaluate fitness
        """

        self.synth.setPatch(individual)
        self.synth.renderPatch()
        out = self.synth.getAudio()
        outFeatures = self.features.getFeatures(out)
        error = AudioEvalBase.absoluteMeanError(self.target, outFeatures)
        return error,


    def predict(self, input):
        """
        Run GA prection on input
        """

        self.target = input

        pop = self.toolbox.population(n=300)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=0.5, mutpb=0.2, ngen=25,
                                       stats=stats, halloffame=hof, verbose=True)

        return hof[0]
