#!/usr/bin/python
"""
Non-dominated Sorting Genetic Algorithm with multiple objectives
"""

import numpy as np
import random
from tqdm import tqdm

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from spiegel.evaluation.evaluation_base import EvaluationBase
from spiegel.estimator.estimator_base import EstimatorBase
from spiegel.synth.synth_base import SynthBase
from spiegel.features.features_base import FeaturesBase


class NSGA3(EstimatorBase):
    """
    """

    def __init__(self, synth, features_list, seed=None):
        """
        Constructor
        """
        super().__init__()

        if not isinstance(synth, SynthBase):
            raise TypeError("synth must be of type SynthBase")

        self.synth = synth
        self.num_params = len(synth.get_patch())

        if not isinstance(features_list, list):
            raise TypeError("features_list must be a list")

        self.features_list = features_list
        self.target = None

        self.logbook = tools.Logbook()

        random.seed(seed)
        self.setup()


    def setup(self):
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
        Evaluate fitness
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
        Run GA prection on input
        """

        self.target = []
        for extractor in self.features_list:
            self.target.append(extractor(input))

        pop = self.toolbox.population(n=300)
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
        pbar = tqdm(range(1, 100), desc="Generation 1")
        for gen in pbar:
            offspring = algorithms.varAnd(pop, self.toolbox, 0.5, 0.5)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from parents and offspring
            pop = self.toolbox.select(pop + offspring, 300)

            # Compile statistics about the new population
            record = stats.compile(pop)
            self.logbook.record(gen=gen, evals=len(invalid_ind), **record)
            pbar.set_description("Generation %s" % gen)

        return tools.selBest(pop, 1)[0]
