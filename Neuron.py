import numpy as np

class Neuron:
    def __init__(self, input_size, output_size): #output_size useful ?
        """
        Creates a Neuron object with random coefficients
        ----
        input:
            input_size : int -> n_c where c is the layer of the neuron
            output_size : int -> n_(c + 1)
        ----
        output:
            void
        """

        self.input_size = input_size
        self.output_size = output_size
        self.beta = np.random.random_sample(input_size + 1) * 2 - 1 #beta_(0...input_size)

    def compute(self, Zc):
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

        assert len(Zc) == self.input_size, "Input of size {} with neuron of input size {}".format(len(Zc), self.input_size)
        linear_comb = self.beta[0] + np.dot(Zc, self.beta[1:]) #sum of beta * Z
        out = 1 / (1 + np.exp(linear_comb))
        return  out
