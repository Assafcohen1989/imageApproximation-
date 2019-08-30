from Gene import Gene
from random import sample, randrange
import numpy as np
import cv2



class Chromosome(object):
    def __init__(self, genes=[], genes_limit=100, image=None):
        if image is None:
            raise Exception("No picture was set as source.")
        self._genes = genes
        self._genes_limit = genes_limit
        self._img = image
        self._id = randrange(10000)

    def __repr__(self):
        return "Chromosome id: {0}, {1} genes.".format(self._id, self.get_size())

    def generate(self, num_of_genes=5):
        if self._genes is None:
            raise Exception("Chromosome already holds genes!.")
        if num_of_genes > self._genes_limit:
            raise Exception("Cannot generate a chromosome with more than {0} genes.".format(self._genes_limit))

        for _ in range(num_of_genes):
            self._genes.append(Gene().generate(self._img.shape[:2]))
        return self

    def get_gene_pool(self):
        return self._genes

    def set_genes_from_list(self, gene_list):
        if gene_list is not None and len(gene_list) > 0:
            for gene in gene_list:
                self._genes.append(gene.copy())
        else:
            raise Exception("Passed gene_list param is empty or None.")

    def add_random_gene(self, parents=None):
        if self.get_size() < self._genes_limit:
            if parents is None:
                self._genes.append(Gene().generate(self._img.shape[:2]))
            else:
                new_gene = sample(parents[0].get_random_genes(num_of_genes=1) + parents[1].get_random_genes(num_of_genes=1), k=1)
                while new_gene in self._genes:
                    new_gene = sample(parents[0].get_random_genes(num_of_genes=1) + parents[1].get_random_genes(num_of_genes=1), k=1)
                self._genes.append(new_gene[0])

    def get_fitness(self, use_opacity=True):
        """
        This function calculates the MSE ('Mean Squared Error') between the  original image
        and the one the chromosome generates.
        The lower the score -> the lower the error -> higher similarity.
        :return: Mean Squared Error between the evaluated picture and the chromosome generated picture.
        """
        chromosome_img = self.generate_chromosome_image(use_opacity=use_opacity)
        fitness = np.sum((chromosome_img.astype("float") - self._img.astype("float")) ** 2)
        fitness /= float(chromosome_img.shape[0] * chromosome_img.shape[1])
        return fitness

    def get_random_genes(self, num_of_genes=1):
        if num_of_genes == 0 or num_of_genes > len(self._genes):
            return []
        return sample(self._genes, k=num_of_genes)

    def get_size(self):
        return len(self._genes)

    def generate_chromosome_image(self, use_opacity=True):
        img = np.zeros((self._img.shape[:2][0], self._img.shape[:2][1], 3), np.uint8)
        for gene in self._genes:
            gene.draw_gene(img, use_opacity=use_opacity)
        return img

    def draw_chromosome(self, use_opacity=True):
        img = self.generate_chromosome_image(use_opacity=use_opacity)
        # Show the results
        cv2.imshow('gene', img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
        return img

