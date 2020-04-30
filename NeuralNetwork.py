from Neuron import Neuron
import numpy as np

class NeuralNetwork:
    """
    fields:
        format: array of integers of size C -> (n_c)_c
        neuron_layers: array of arrays ofNeurons array length C. The c th layer has length n_c
            -> a line represents a layer of neurons
        Z_layers: float array array -> stores the current output of each Neurons
        learning_Rate: float -> learning Rate used for backpropagation
        current_input: float array of size p -> the input being computed   #Useful ?
        errors: float array of size n_C -> the error of each line
    """

    def __init__(self, format, p):
        """
        Creates a NeuralNetwork object with random coefficients and a given
        size for each layer
        ----
        input:
            format: array of integer of size C -> (n_c)_c
            p: int -> the numberof column used as input of the network
        ----
        output:
            void
        """

        self.format = format
        C = len(format)
        format_include_input = [p] + format
        self.neuron_layers = [[Neuron(format_include_input[layer]) for _ in range(format[layer])] for layer in range(0, C)]
        self.Z_layers = [np.array([0 for _ in range(format[layer + 1])]) for layer in range(0, C)]
        self.learning_Rate = 0.5 #Arbitrary
        self.current_input = [0 for _ in range(p)]
        self.errors = [0 for _ in range(format[-1])]

    def compute_one(self, input):
        """
        Make the input go through the network and stores the outputs of each layer
        ----
        input:
            input: array of floats of size p -> (x_i)_i
        ----
        output:
            void
        """

        current_input = input[:]
        C = len(self.format)
        n0 = self.format[0]
        for j in range(n0):
            neuron = self.neuron_layers[0][j]
            self.Z_layers[0][j] = neuron.compute_output(input)

        for c in range(1, C):
            nc = self.format[c]
            layer_input = self.Z_layers[c - 1]
            for j in range(nc):
                neuron = self.neuron_layers[c][j]
                self.Z_layers[c][j] = neuron.compute_output(layer_input)


    def compute_all(self, database):
        """
        Make the every line go through the network, storing the errors of each one
        ----
        input:
            database: array of shape (N, p) -> the training database
        ----
        output:
            void
        """

        pass

    def errori(self, i):
    """
    Returns the error of one line
    ----
    input:
        i: int -> index of the line
    ----
    output:
        res: float -> R_i(theta)
    """

        pass

    def error():
        """
        Returns the error of one line
        ----
        input:
            void
        ----
        output:
            res: float -> R = \sum_i R_i
        """

        pass

    def deriv(i, j, k, c):
        """
        Returns the derivative of R_i with respect to beta_(i, j)^k
        ----
        input:
            i, j, k, c: int
        ----
        output:
            res: float -> the derivative
        """

        pass

    def update_coeff():
        """
        Update every coefficient of the network using backpropagation
        ----
        input:
            void
        ----
        output:
            void
        """
        pass
