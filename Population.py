import os, cv2
from argparse import ArgumentParser
from functools import reduce
from random import randrange, sample, uniform

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

    @property
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

    def _calculate_scores(self):
        if self._chromosomes is None or len(self._chromosomes) == 0:
            return 0
        for chromosome in self._chromosomes:
            chromosome['score'] = chromosome['chromosome'].get_fitness()

    def get_population_fitness(self):
        self._calculate_scores()
        score = sum([c['score'] for c in self._chromosomes]) / len(self._chromosomes)
        print("Population calculated fitness is {0}".format(score))
        return score

    def _selection(self, size=2):
        self._calculate_scores()
        return sorted(self._chromosomes,
                      key=lambda x: x['score'])[:size]  # New parents for the next generation

    def _crossover(self, parents=None, number_of_offsprings=10):
        if parents is None:
            raise Exception("Skipped selection stage.\nNo parents were selected.")

        offsprings = []
        parent_1 = parents[0]
        parent_2 = parents[1]
        new_offspring_size = max(parent_1.get_size(), parent_2.get_size())

        for _ in range(number_of_offsprings):
            split = randrange(1, new_offspring_size)
            offsprings.append(
                Chromosome(
                    genes=parent_1.get_random_genes(num_of_genes=split) + parent_2.get_random_genes(num_of_genes=new_offspring_size - split),
                    genes_limit=100,
                    image=self._img))

        return offsprings

    def _mutate(self, chromosomes=None, p_grow=0.2):
        if chromosomes is None:
            raise Exception("Nothing to mutate.\nNo chromosomes were given.")

        # random number of chromosomes to mutate
        chromosomes_to_mutate = sample(chromosomes, k=randrange(len(chromosomes) + 1))
        for chromosome in chromosomes_to_mutate:
            if uniform(0, 1) <= p_grow:
                chromosome.add_random_gene()
            # random number of genes to mutate
            genes_to_mutate = chromosome.get_random_genes(num_of_genes=randrange(chromosome.get_size()))
            for gene in genes_to_mutate:
                gene.mutate()

    def breed(self):
        i = 0
        while True:
            parents = self._selection()
            if i%1000 == 0:
                parents[0]['chromosome'].draw_chromosome()
                print("Iteration: {0}".format(i))
            offsprings = self._crossover([p['chromosome'] for p in parents])
            self._mutate(offsprings)
            self._chromosomes = [{'chromosome': offspring, 'score': 0} for offspring in offsprings]
            i+=1
        # return Population(chromosome_list=offsprings)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--pic", required=True, help="--pic - path to the requested picture file.")
    args = parser.parse_args()
    return args


def main():
    file_path = get_args().pic
    if not os.path.isfile(file_path):
        raise FileExistsError("Could not find requested file.")
    img = cv2.imread(file_path)
    pop = Population([], 10, img)
    pop.create_population()

    pop.breed()


if __name__ == '__main__':
    main()
