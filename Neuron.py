import numpy as np

class Neuron:
    """
    fields:
        input_size: int
        beta: list of the coefficitents used in the linear combination of the
            outputs of the previous layer
    """

    def __init__(self, input_size):
        """
        Creates a Neuron object with random coefficients
        ----
        input:
            input_size: int -> n_c where c is the layer of the neuron
        ----
        output:
            void
        """

        self.input_size = input_size
        self.beta = np.random.random_sample(input_size) * 2 - 1 #beta_(1...input_size)

    def comb_lin(self, Zc):
        """
        Returns the sum of beta * Z
        ----
        Input:
            Zc: array of length n_c -> outputs of the previous layer
        ----
        output:
            out: float -> sum of beta * Z
        """

        assert len(Zc) == self.input_size, "Input of size {} with neuron of input size {}".format(len(Zc), self.input_size)
        return np.dot(Zc, self.beta) #sum of beta * Z



    def compute_output(self, Zc):
        """
        Given the outputs of the previous layer, computes the output of
        this neuron
        ----
        input:
            Zc : array of length n_c -> outputs of the previous layer
        ----
        output:
            out : float -> output of the neuron
        """

        linear_comb = self.comb_lin(Zc)
        out = 1. / (1. + np.exp(- linear_comb))
        return  out
