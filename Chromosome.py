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

    def generate(self, num_of_genes=3):
        if self._genes is None:
            raise Exception("Chromosome already holds genes!.")
        if num_of_genes > self._genes_limit:
            raise Exception("Cannot generate a chromosome with more than {0} genes.".format(self._genes_limit))

        for _ in range(num_of_genes):
            self._genes.append(Gene().generate(self._img.shape[:2]))
        return self

    def set_genes_from_list(self, gene_list):
        if gene_list is not None and len(gene_list) > 0:
            self._genes = gene_list
        else:
            raise Exception("Passed gene_list param is empty or None.")

    def add_random_gene(self):
        if self.get_size() < self._genes_limit:
            self._genes.append(Gene().generate(self._img.shape[:2]))

    def get_fitness(self):
        """
        This function calculates the MSE ('Mean Squared Error') between the  original image
        and the one the chromosome generates.
        The lower the score -> the lower the error -> higher similarity.
        :return:
        """
        chromosome_img = self.generate_chromosome_image()
        fitness = np.sum((chromosome_img.astype("float") - self._img.astype("float")) ** 2)
        fitness /= float(chromosome_img.shape[0] * chromosome_img.shape[1])
        return fitness
    '''
    Need to calculate the current chromosome's similarity to the original picture
    '''

    def get_random_genes(self, num_of_genes=1):
        return sample(self._genes, k=num_of_genes)

    def get_size(self):
        return len(self._genes)

    def generate_chromosome_image(self):
        img = np.zeros((self._img.shape[:2][0], self._img.shape[:2][1], 3), np.uint8)
        for gene in self._genes:
            gene.draw_gene(img)
        return img

    def draw_chromosome(self):
        img = self.generate_chromosome_image()
        # Show the results
        cv2.imshow('gene', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return img
