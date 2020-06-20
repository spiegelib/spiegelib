#!/usr/bin/python
"""
Non-dominated Sorting Genetic Algorithm with multiple objectives. Based on the
implementation by Tatar et al. [1]_
"""

import random

import numpy as np
from tqdm import tqdm
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from spiegelib.evaluation.evaluation_base import EvaluationBase
from spiegelib.estimator.estimator_base import EstimatorBase
from spiegelib.synth.synth_base import SynthBase
from spiegelib.features.features_base import FeaturesBase


class NSGA3(EstimatorBase):
    """
        Args:
            synth (Object): Instance of :class:`~spiegelib.synth.SynthBase`
            features (list): A list of :class:`~spiegelib.features.FeaturesBase` objects.
                Each feature extraction object defines an objective and is used
                in the evaluation function to determine *fitness* of an individual.
            seed (int, optional): Seed for random. Defaults to current system time.
            pop_size (int, optional): Size of population at each generation
            ngen (int, optional): Number of generations to run
            cxpb (float, optional): Crossover probability, must be between 0 and 1.
            mutpb (float, optional): Mutation probability, must be between 0 and 1.
    """

    def __init__(self, synth, features, seed=None, pop_size=100, ngen=25,
                 cxpb=0.5, mutpb=0.5):
        """
        Constructor
        """
        super().__init__()

        if not isinstance(synth, SynthBase):
            raise TypeError("synth must be of type SynthBase")

        self.synth = synth
        self.num_params = len(synth.get_patch())

        if not isinstance(features, list):
            raise TypeError("features_list must be a list")

        self.features_list = features
        self.target = None

        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb

        self.logbook = tools.Logbook()

        random.seed(seed)
        self._setup()


    def _setup(self):
        """
        Setup genetic algorithm
        """

        self.num_objectives = len(self.features_list)

        # Create individual types
        creator.create("FitnessMin", base.Fitness,
                       weights=(-1.0,) * self.num_objectives)
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
            self.num_params
        )
        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        ref_points = tools.uniform_reference_points(self.num_objectives, 12)

        self.toolbox.register("evaluate", self.fitness)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0,
                              up=1.0, eta=30.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0,
                              up=1.0, eta=20.0, indpb=1.0/self.num_params)
        self.toolbox.register("select", tools.selNSGA3, ref_points=ref_points)


    def fitness(self, individual):
        """
        This is automatically called during prediction. Evaluation that calculates
        the fitness of an individual. The individual is a new estimated synthesizer
        parameter setting, and fitness is calculated by rendering an audio sample using
        those parameter settings and then measuring the error between that sample
        and the target sound using a set of audio feature extractors set during
        construction.

        Args:
            individual (list): List of float values representing a synthesizer patch

        Returns:
            list: A list of the error values, one for each feature extractor
        """

        self.synth.set_patch(individual)
        self.synth.render_patch()
        out = self.synth.get_audio()

        errors = []
        index = 0
        for extractor in self.features_list:
            out_features = extractor(out)
            errors.append(EvaluationBase.mean_abs_error(self.target[index],
                                                        out_features))
            index += 1

        return errors


    def predict(self, input):
        """
        Run prection on input audio target

        Args:
            input (:ref:`AudioBuffer <audio_buffer>`): AudioBuffer to use as target
        """

        self.target = []
        for extractor in self.features_list:
            self.target.append(extractor(input))

        pop = self.toolbox.population(n=self.pop_size)
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        self.logbook.header = "gen", "evals", "std", "min", "avg", "max"

        record = stats.compile(pop)
        self.logbook.record(gen=0, evals=len(invalid_ind), **record)

        # Begin the generational process
        pbar = tqdm(range(1, self.ngen + 1), desc="Generation 1")
        for gen in pbar:
            offspring = algorithms.varAnd(pop, self.toolbox, self.cxpb, self.mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            pop = self.toolbox.select(pop + offspring, self.pop_size)

            # Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            pbar.set_description("Generation %s" % gen)

        return tools.selBest(pop, 1)[0]
