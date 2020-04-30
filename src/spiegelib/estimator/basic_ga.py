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

from spiegelib.evaluation.evaluation_base import EvaluationBase
from spiegelib.estimator.estimator_base import EstimatorBase
from spiegelib.synth.synth_base import SynthBase
from spiegelib.features.features_base import FeaturesBase


class BasicGA(EstimatorBase):
    """
    Args:
        synth (Object): Instance of :class:`spiegelib.synth.SynthBase`
        features (Object): Instance of :class:`spiegelib.features.FeaturesBase`.
            This is used in the evaluation function to determine *fitness* of an individual.
        seed (int, optional): Seed for random. Defaults to current system time.
        pop_size (int, optional): Size of population at each generation
        ngen (int, optional): Number of generations to run
        cxpb (float, optional): Crossover probability, must be between 0 and 1.
        mutpb (float, optional): Mutation probability, must be between 0 and 1.
    """

    def __init__(self, synth, features, seed=None, pop_size=100, ngen=25,
                 cxpb=0.5, mutpb=0.3):
        """
        Constructor
        """
        super().__init__()

        if not isinstance(synth, SynthBase):
            raise TypeError("synth must be of type SynthBase")

        self.synth = synth
        self.num_params = len(synth.get_patch())

        if not isinstance(features, FeaturesBase):
            raise TypeError("features must be of type FeaturesBase")

        self.features = features
        self.target = None

        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb

        self.logbook = None

        random.seed(seed)
        self._setup()


    def _setup(self):
        """
        Setup genetic algorithm
        """

        # Create individual types
        creator.create("BasicGAFitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("BasicGAIndividual", list, fitness=creator.BasicGAFitnessMin)

        # Setup toolbox
        self.toolbox = base.Toolbox()

        # Attribute generator
        self.toolbox.register("attr_float", random.random)

        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.BasicGAIndividual,
                              self.toolbox.attr_float, self.num_params)

        self.toolbox.register("population", tools.initRepeat, list,
                              self.toolbox.individual)

        self.toolbox.register("evaluate", self.fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)


    def fitness(self, individual):
        """
        This is automatically called during prediction. Evaluation that calculates
        the fitness of an individual. The individual is
        a new estimated synthesizer parameter setting, and fitness is calculated
        by rendering an audio sample using those parameter settings and then measuring
        the error between that sample and the target sound using audio feature extraction.
        The type of feature extraction is set in the constructor.

        Args:
            individual (list): List of float values representing a synthesizer patch

        Returns:
            list: A list of the error values (b/c this is a single objective GA \
                the list only has one element)
        """

        self.synth.set_patch(individual)
        self.synth.render_patch()
        out = self.synth.get_audio()
        out_features = self.features(out)
        error = EvaluationBase.mean_abs_error(self.target, out_features)
        return error,


    def predict(self, input):
        """
        Run GA prection on input audio target

        Args:
            input (:ref:`AudioBuffer <audio_buffer>`): AudioBuffer to use as target
        """


        self.target = self.features(input)

        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, self.logbook = algorithms.eaSimple(pop, self.toolbox, cxpb=self.cxpb,
                                                mutpb=self.mutpb, ngen=self.ngen,
                                                stats=stats, halloffame=hof,
                                                verbose=True)

        return hof[0]
