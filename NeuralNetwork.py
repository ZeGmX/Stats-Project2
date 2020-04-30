from Neuron import Neuron

class NeuralNetwork:
    """
    fields:
        format: array of integers of size C -> (n_c)_c
        neuron_layer: array of arrays ofNeurons array length C. The c th layer has length n_c
            -> a line represents a layer of neurons
        Z_layer: float array array -> stores the current output of each Neurons
        learning_Rate: float -> learning Rate used for backpropagation
        current_input: float array of size p -> the input being computed
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
        self.neuron_layer = [[Neuron(format_include_input[layer]) for _ in range(format[layer])] for layer in range(0, C)]
        self.Z_layer = [[0 for _ in range(format[layer + 1])] for layer in range(0, C)]
        self.learning_Rate = 0.5 #Arbitrary
        self.current_input = [0 for _ in range(p)]
