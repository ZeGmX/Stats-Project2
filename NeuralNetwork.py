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
        derivatives: float array -> derivatives[c][k][j] = dR / dbeta_(j, k)^c
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
        self.neuron_layers = [[Neuron(format_include_input[layer]) for _ in range(format[layer])] for layer in range(C)]
        self.Z_layers = [np.array([0 for _ in range(format[layer])]) for layer in range(C)]
        self.learning_Rate = 0.5 #Arbitrary
        self.current_input = [0 for _ in range(p)] #Useful ?
        self.errors = [0 for _ in range(format[-1])]
        self.derivatives = [[[0 for _ in range(format_include_input[layer])] for _ in range(format[layer])] for layer in range(C)]

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


    def compute_all(self, database, outputs):
        """
        Make the every line go through the network, storing the errors of each one
        ----
        input:
            database: array of shape (N, p) -> the training database
            outputs: array of shape (N, n_C) -> the expected outputs (y)
        ----
        output:
            void
        """

        for i in range(len(database)):
            X_i = database[i]
            Y_i = outputs[i]
            self.compute_one(X_i)
            R_i = self.compute_error(Y_i)
            self.errors[i] = R_i


    def compute_error(self, expected_output):
    """
    Returns the current error
    ----
    input:
        expect_output: float array of size n_C -> y
    ----
    output:
        res: float -> R_i(theta)
    """

        last_output = self.Zlines[-1]
        Y_hat = - np.log(1 / last_output - 1) #sigma shouldn't be used for the last line
        res = sum((Y_hat - expected_output) ** 2)
        return res

    def error(self):
        """
        Returns the error of one line
        ----
        input:
            void
        ----
        output:
            res: float -> R = \sum_i R_i
        """

        return sum(self.errors)

    def deriv(self, i, j, k, c):
        """
        Returns the derivative of R_i with respect to beta_(j, k)^c
        ----
        input:
            i, j, k, c: int
        ----
        output:
            res: float -> the derivative
        """

        pass #TODO

    def compute_derivatives(self):
        """
        Stores the derivatives of R with respect of every coefficient
        We need that because the change of a coefficient should not impact the
        change of another one
        ----
        input:
            void
        ----
        output:
            void
        """

        C = len(self.format)
        for c in range(C):
            nc = len(self.neuron_layers[c][0].beta)
            ncp1 = self.format[c]
            for k in range(ncp1):
                neuron = self.neuron_layers[c][k]
                for j in range(nc):
                    self.derivatives[c][k][j] = sum(self.deriv(i, j, k, c) for i in range(out_size))


    def update_coeff(self):
        """
        Update every coefficient of the network using backpropagation
        ----
        input:
            void
        ----
        output:
            void
        """

        C = len(self.format)
        out_size = len(self.errors)
        self.compute_derivatives()
        for c in range(C):
            nc = len(self.neuron_layers[c][0].beta)
            ncp1 = self.format[c]
            for k in range(ncp1):
                neuron = self.neuron_layers[c][k]
                for j in range(nc):
                    neuron.beta[j] -= self.learning_Rate * self.derivatives[c][k][j] #beta_(j, k)^c = beta_(j, k)^c - eta * dR/dbeta_(j, k)^c
