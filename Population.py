import cv2
import os
import time
from argparse import ArgumentParser
from random import randrange, sample, uniform
import signal
from contextlib import contextmanager

from Chromosome import Chromosome


class Population(object):
    def __init__(self, chromosome_list: list = None, population_size_limit: int = 10, image=None):
        if chromosome_list is None:
            chromosome_list = []
        if image is None:
            raise Exception("No picture was set as source.")
        self._chromosomes = chromosome_list
        self._size_limit = population_size_limit
        self._img = image
        self.__last_parents = []

    def __repr__(self):
        pop = "Population: "
        for chromosome in self._chromosomes:
            pop += chromosome.__repr__ + "\n"
        return pop

    def create_population(self, initial_size=10):
        if self._chromosomes is None:
            raise Exception("Population already initialised!.")

        if initial_size > self._size_limit:
            raise Exception("Requested population is too big.\nMaximum size is {0}".format(self._size_limit))

        for i in range(initial_size):
            self._chromosomes.append({'chromosome': Chromosome([], 100, self._img).generate(), 'score': 0})

    def _calculate_scores(self, use_opacity=True):
        if self._chromosomes is None or len(self._chromosomes) == 0:
            return 0
        for chromosome in self._chromosomes:
            chromosome['score'] = chromosome['chromosome'].get_fitness(use_opacity=use_opacity)

    def _selection(self, size=2):
        self._calculate_scores(use_opacity=True)
        parents = sorted(self._chromosomes, key=lambda x: x['score'], reverse=False)
        print("Parent1({2},{4}) score: {0}, Parent2({3},{5}) score: {1}".format(parents[0]['score'], parents[1]['score'],
                                                                                parents[0]['chromosome']._id, parents[1]['chromosome']._id,
                                                                                parents[0]['chromosome'].get_size(), parents[1]['chromosome'].get_size()))
        return parents[:size]  # New parents for the next generation

    def _crossover(self, parents=None, number_of_offsprings=50, p_grow=0.5):
        if parents is None:
            raise Exception("Skipped selection stage.\nNo parents were selected.")

        parent_1 = parents[0]
        parent_2 = parents[1]
        offsprings = [parent_1, parent_2]

        for _ in range(number_of_offsprings-2):
            grow = 0
            if uniform(0, 1) <= p_grow:
                grow = randrange(1, 4)
            half_of_parent_1 = parent_1.get_random_genes(num_of_genes=int(parent_1.get_size()/2) + grow)
            half_of_parent_2 = parent_2.get_random_genes(num_of_genes=int(parent_2.get_size()/2))
            son = Chromosome(
                    genes=[],
                    genes_limit=100,
                    image=self._img)

            son.set_genes_from_list(half_of_parent_1 + half_of_parent_2)
            offsprings.append(son)

        return offsprings

    def _mutate(self, chromosomes=None, p_grow=0.5, p_chromosomes_to_change=0.2, divergence=False):
        if chromosomes is None:
            raise Exception("Nothing to mutate.\nNo chromosomes were given.")

        # random number of chromosomes to mutate
        chromosomes_to_mutate = sample(chromosomes[2:], k=int((len(chromosomes) - 2) * p_chromosomes_to_change))
        for chromosome in chromosomes_to_mutate:
            chance_to_grow = uniform(0, 1)
            if chance_to_grow <= p_grow:
                chance = uniform(0, 1)
                if 0 < chance <= 0.5:
                    chromosome.add_random_gene()  # Adding a completely random gene
                    if divergence:
                        chromosome.add_random_gene()  # Adding a completely random gene

                elif 0.5 < chance <= 0.8:
                    chromosome.add_random_gene(chromosomes[:2])  # Adding a random non existing gene from one of the parents
                else:
                    chromosome.add_random_gene(chromosome)  # Duplicating a present gene which wil probably mutate

            # random number of genes to mutate
            else:
                chromosome_size = chromosome.get_size()
                if chromosome_size == 1:
                    genes_to_mutate = chromosome.get_random_genes(num_of_genes=1)
                else:
                    genes_to_mutate = chromosome.get_random_genes(num_of_genes=randrange(1, chromosome_size))

                for gene in genes_to_mutate:
                    if divergence:
                        gene.mutate(num_of_mutations=randrange(2), step=0.3)
                    else:
                        gene.mutate(num_of_mutations=1, step=0.2)

    def breed(self, time_limit_in_minuets=5):
        start_time = time.time()
        i = 0
        last_parents = self._selection()
        while True:
            parents = self._selection()
            if i % 10 == 0:
                parents[0]['chromosome'].draw_chromosome(use_opacity=True)
            print("Iteration: {0}".format(i))
            divergence_flag = False
            if parents == last_parents:
                divergence_flag = True

            offsprings = self._crossover(parents=[p['chromosome'] for p in parents], number_of_offsprings=30, p_grow=0.5)
            self._mutate(offsprings, p_grow=0.5, p_chromosomes_to_change=0.4, divergence=divergence_flag)
            self._chromosomes = [{'chromosome': offspring, 'score': 0} for offspring in offsprings]
            i += 1
            end_time = time.time()
            if end_time - start_time > time_limit_in_minuets * 60:
                print("Reached time limit of {0} minuets.".format(time_limit_in_minuets))
                break
        return parents[0]['chromosome']


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pic", required=True, help="--pic - path to the requested picture file.")
    parser.add_argument("--timeout", required=False, default=5, help="--timeout - Limit time for running in minuets.")
    args = parser.parse_args()
    return args


def main():
    print("Initializing Image approximation using Genetic Algorithm")
    file_path = get_args().pic
    if not os.path.isfile(file_path):
        raise FileExistsError("Could not find requested file.")
    img = cv2.imread(file_path)
    print("Loaded image successfully.")
    pop = Population([], 10, img)
    pop.create_population()
    print("Initial population initialized.")
    timeout = get_args().timeout
    print("Starting breeding with time limit of {0} minuets.".format(timeout))
    result = pop.breed(time_limit_in_minuets=int(timeout))
    print("Displaying final result.")
    result.draw_chromosome(use_opacity=True, hold=True)


if __name__ == '__main__':
    main()
