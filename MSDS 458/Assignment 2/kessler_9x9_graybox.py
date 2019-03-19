# -*- coding: utf-8 -*-
"""Training a simple Neural Network using numpy - Alan Kessler

Assumptions:
    * Assumes sigmoid transfer function
    * Assumes a single layer of hidden nodes
    * Assumes batch size is fixed at 1

"""

import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

def process_data():
    """Process data from orignal script

    Big Classes:
        Line in the Middle ('A'): A, B, E, F, H, P, R, S
        Mostly Circular ('C'): C, D, G, O, Q, U
        Diagonal Present ('J'): J, K, M, N, W, X, Z
        Other ('I'): I, L, T

    """

    # List of data from the 2017-03-11 File with modified indices
    raw_alpha = [(0,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],1,'A',0,'A'),
                 (1,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],2,'B',0,'A'),
                 (2,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],3,'C',1,'C'),
                 (3,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],4,'D',1,'C'),
                 (4,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],5,'E',0,'A'),
                 (5,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],6,'F',0,'A'),
                 (6,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],7,'G',1,'C'),
                 (7,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],8,'H',0,'A'),
                 (8,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1,0,0],9,'I',3,'I'),
                 (9,[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0,0,0],10,'J',2,'J'),
                 (10,[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0],11,'K',2,'J'),
                 (11,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],12,'L',3,'I'),
                 (12,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1],13,'M',2,'J'),
                 (13,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1, 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1],14,'N',2,'J'),
                 (14,[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1,0],15,'O',1,'C'),
                 (15,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],15,'P',0,'A'),
                 (16,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,1],17,'Q',1,'C'),
                 (17,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],15,'R',0,'A'),
                 (18,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],19,'S',0,'A'),
                 (19,[0,1,1,1,1,1,1,1,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],20,'T',3,'I'),
                 (20,[1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 0,1,1,0,0,0,1,1,0, 0,0,1,1,1,1,1,0,0],21,'U',1,'C'),
                 (21,[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0],23,'W',2,'J'),
                 (22,[1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],24,'X',2,'J'),
                 (23,[1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,],26,'Z',2,'J')]

    # List of indices to be one-hot encoded
    # 1 - For Letters, 2 - For Big Classes
    indices_1 = []
    indices_2 = []

    # Initial array to load training data input
    training_data = np.zeros((len(raw_alpha), 81))

    # List of letter labels
    letters = []

    # Loop through letters to process data
    for i in raw_alpha:
        indices_1.extend([i[0]])
        indices_2.extend([i[4]])
        training_data[i[0]] = i[1]
        letters.extend([i[3]])

    # One hot encode letters
    indices_1_array = np.array(indices_1)
    labels_1 = np.zeros((24, 24))
    labels_1[np.arange(24), indices_1_array] = 1

    indices_2_array = np.array(indices_2)
    labels_2 = np.zeros((24, 4))
    labels_2[np.arange(24), indices_2_array] = 1

    return labels_1, labels_2, training_data, letters

def randomize_input(training_labels_1, training_labels_2, training_data,
                    letters, gamma=0.2, num_variants=10):
    """Helper function to create new variants specifically for this data"""

    # Start with copying the input data
    full_training_data = np.copy(training_data)
    full_training_labels_1 = np.copy(training_labels_1)
    full_training_labels_2 = np.copy(training_labels_2)
    full_letters = np.copy(letters)

    # Create a variant by switching each pixel with probability gamma
    for i in range(0, num_variants):
        modified_data = np.copy(training_data)
        for ob_num, record in enumerate(modified_data):
            for pixel_num, pixel in enumerate(record):
                if pixel == 0:
                    modified_data[ob_num][pixel_num] = np.random.choice(2, p=[1-gamma, gamma])
                else:
                    modified_data[ob_num][pixel_num] = np.random.choice(2, p=[gamma, 1-gamma])
        # Add records to the arrays
        full_training_data = np.concatenate((full_training_data, modified_data), axis=0)
        full_training_labels_1 = np.concatenate((full_training_labels_1, training_labels_1), axis=0)
        full_training_labels_2 = np.concatenate((full_training_labels_2, training_labels_2), axis=0)
        full_letters = np.concatenate((full_letters, letters), axis=0)

    return full_training_labels_1, full_training_labels_2, full_training_data, full_letters



class NeuralNet:
    """Describes a neural network of the type used in this example"""

    def __init__(self, size, alpha=1, seed=9798, fit=None):
        """A simple 3-layer network definition

        Args:
            size: size: tuple of input size, hidden nodes, and output classes
            alpha: steepness parameter (default 1)
            seed: random number seed (default 9798)
        Raises:
            TypeError: if alpha, size, or seed are incorrectly formatted
            ValueError: if size is of an incorrect dimension

        """
        # Check input types
        if not isinstance(alpha, (int, float)):
            raise TypeError("alpha (steepness) must be type int or float")
        if not isinstance(size, (tuple)):
            raise TypeError("size must be a tuple of size 3")
        if not isinstance(seed, (int)):
            raise TypeError("seed must be an integer")

        if not len(size) == 3:
            raise ValueError("size must be a tuple of size 3")

        self.size = size
        self.alpha = alpha
        np.random.seed(seed)
        self.fit = fit

    def sigmoid_transfer(self, net_sum):
        """Calculate the sigmoid transfer function.

        Args:
            net_sum: the input into the node.
        Returns:
            the activation output of the node.
        Raises:
            TypeError: if net_sum is not a number.

        """
        return 1 / (1 + np.exp(-net_sum * self.alpha))

    def initial_weights(self):
        """Randomly assigns initial weights [-1, 1)

        Returns:
            list of initial weight arrays

        """
        input_to_hidden = np.random.uniform(-1, 1,
                                            (self.size[1], self.size[0]))
        hidden_to_output = np.random.uniform(-1, 1,
                                             (self.size[2], self.size[1]))
        return [input_to_hidden, hidden_to_output]

    def initial_biases(self):
        """Randomly assigns initial biases [-1, 1)

        Returns:
            list of initial weight arrays

        """
        input_to_hidden = np.random.uniform(-1, 1, self.size[1])
        hidden_to_output = np.random.uniform(-1, 1, self.size[2])
        return [input_to_hidden, hidden_to_output]

    def forward_pass(self, case_input, weights=None, biases=None):
        """Calculate the result of a forward pass through the neural network

        Args:
            case_input: array of input into the neural network
            weights: list of network weight arrays (default None)
            biases: list of network bias arrays (default None)
        Returns:
            a neural network fit containing results
        Raises:
            TypeError: if arguments are not arrays.

        """

        # If not supplied, generate random weights
        if weights is None:
            weights = self.initial_weights()
        if biases is None:
            biases = self.initial_biases()

        # Check input types
        if not isinstance(case_input, np.ndarray):
            raise TypeError("case_input must be an array")
        if not isinstance(weights[0], np.ndarray):
            raise TypeError("weights must be a list of arrays")
        if not isinstance(biases[0], np.ndarray):
            raise TypeError("biases must be a list of arrays")

        # Calculate the summed input to the hidden layer
        net0 = np.matmul(case_input, np.transpose(weights[0])) + biases[0]

        # Apply sigmoid transfer function
        out0 = self.sigmoid_transfer(net0)

        # Calculate the summed input to the output layer
        net1 = np.matmul(out0, np.transpose(weights[1])) + biases[1]

        # Apply sigmoid transfer function
        out1 = self.sigmoid_transfer(net1)

        # Apply basic credit assignment to the pixels
        credit = case_input * weights[0]

        # Store results
        forpass = collections.namedtuple('ForwardPass',
                                         ['case_input', 'size', 'alpha',
                                          'weights', 'biases', 'net0', 'out0',
                                          'net1', 'out1', 'credit'])

        self.fit = forpass(case_input, self.size, self.alpha, weights, biases,
                           net0, out0, net1, out1, credit)

        return self

    def sse_eval(self, target):
        """Calculate the sum of squared errors for a neural network

        Args:
            target: array of ideal output
        Returns:
            sum of squared errors from the network
        Raises:
            TypeError: if target is not an array

        """

        # Check input type
        if not isinstance(target, np.ndarray):
            raise TypeError("target must be an array")

        sse = np.sum((target - self.fit.out1)**2)

        return sse

    def backpropagation(self, target, eta=0.5):
        """Conducts a step of backpropagation

        Args:
            target: array of ideal output
            eta: learning rate (default 0.5)
        Returns:
            results of backpropagation
        Raises:
            TypeError: if arguments are not an array, NeuralNet, and numeric.
            ValueError: if eta is not bounded (0, 1]

        """

        # Check if eta is a number
        if not isinstance(eta, (int, float)):
            raise TypeError("eta (learning rate) must be type int or float")

        if eta <= 0 or eta > 1:
            raise ValueError("eta (learning rate) must be bounded (0, 1]")

        # Working backwards so delta0 is array of deltas closest to output layer
        delta0 = -self.alpha*(target - self.fit.out1)*self.fit.out1*(1 - self.fit.out1)

        # Adjusted weights from hidden to output layer
        weights0 = self.fit.weights[1] - eta*np.outer(delta0, self.fit.out0)

        # Adjust bias from hidden to output layer
        biases0 = self.fit.biases[1] - eta*delta0

        # Working backwards so delta1 is array of deltas closest to input layer
        summation = np.sum(np.transpose(self.fit.weights[1])*delta0, axis=1)
        delta1 = summation*self.fit.out0*(1 - self.fit.out0)*self.alpha

        # Adjusted weights from input to hidden layer
        weights1 = self.fit.weights[0] - eta*np.outer(delta1, self.fit.case_input)

        # Adjust bias from input to hidden layer
        biases1 = self.fit.biases[0] - eta*delta1

        # Create a new weights array containing all weights
        weights_new = [weights1, weights0]

        # Create a new biases array containing all biases
        biases_new = [biases1, biases0]

        # Revise the fit with the new weights and biases
        self.forward_pass(self.fit.case_input, weights=weights_new,
                          biases=biases_new)

        return self

    def train(self, data_set, target, eta=0.5, epsilon=0.02, iterations=5000, verbose=True):
        """Trains the neural network and predicts target

        Assumptions:
            Each iteration (epoch) uses all input records in random order

        Args:
            data_set: array of input data (multiple observations)
            target: array of target output (multiple observations)
            eta: learning rate (default 0.5)
            epsilon: early stopping criteria ASE (default 0.02)
            iterations: number of maximum training iterations (default 5000)
            verbose: if true, print results (default True)
        Returns:
            a trained neural network
        Raises:
            TypeError: if epsilon is not a float.
            ValueError: if the number of iterations is less than or equal to 0.

        """

        # Check if epsilon a float
        if not isinstance(epsilon, float):
            raise TypeError("epsilon (learning rate) must be type float")

        # Check iterations value
        if iterations <= 0:
            raise ValueError("number of iterations must be greater than 0")

        # Calculate a baseline SSE (random for each)
        sse = 0
        for i in range(0, data_set.shape[0]):
            data_single = data_set[i]
            target_single = target[i]
            self.forward_pass(data_single)
            sse += self.sse_eval(target_single)
            ase = sse / data_set.shape[0]

        if verbose:
            print("Training Model")
            print("\nInput:")
            print(data_set)
            print("\nTarget:")
            print(target)
            print("\n Average Squared Error by Iteration")
            print(f"\n{0:4}. {sse:.4f}")

        # Train over the specified number of iterations
        for iteration_num in range(1, iterations+1):
            # Perform backpropagation for each training record
            shuffle = np.random.choice(data_set.shape[0], data_set.shape[0],
                                       replace=False)
            for i in shuffle:
                data_single = data_set[i]
                target_single = target[i]
                self.forward_pass(data_single, weights=self.fit.weights,
                                  biases=self.fit.biases)
                self.backpropagation(target_single, eta=eta)

            # Calculate total SSE for the iteration
            sse = 0
            for i in range(0, data_set.shape[0]):
                data_single = data_set[i]
                target_single = target[i]
                self.forward_pass(data_single, weights=self.fit.weights,
                                  biases=self.fit.biases)
                sse += self.sse_eval(target_single)
            # Convert SSE to Average Squared Error
            ase = sse / data_set.shape[0]
            if verbose and iteration_num % 10 == 0:
                print(f"{iteration_num:4}. {ase:.4f}")

            # Enable early stopping past the threshold set
            if ase <= epsilon or iteration_num == iterations:
                if verbose:
                    for i in range(0, data_set.shape[0]):
                        data_single = data_set[i]
                        target_single = target[i]
                        self.forward_pass(data_single, weights=self.fit.weights,
                                          biases=self.fit.biases)
#                         print("\n------------")
#                         print("\nInput array:")
#                         print(self.fit.case_input)
#                         print("\nTarget array")
#                         print(target_single)
#                         print("\nHidden Layer Output:")
#                         print(self.fit.out0)
#                         print("\nOutput Layer Final Output:")
#                         print(self.fit.out1)
                break

        return self

def display_letter(case_input_single):
    """Quick display of letter (assuming square)"""

    # Dimension assuming square
    img_size = np.sqrt(case_input_single.shape[0]).astype(int)

    # Reshape assumping square
    img = case_input_single.reshape(img_size, img_size)

    # Select white background with blue letters
    colors = np.array([[1, 1, 1],
                       [0, 0, 1]])
    cmap = matplotlib.colors.ListedColormap(colors)

    # Show image
    plt.imshow(img, cmap=cmap)
    plt.show()

def display_node(data_set, model, node_num):
    """Display of single hidden layer activation"""

    # Select white background with gray letters
    colors = np.array([[1, 1, 1],
                       [0.9, 0.9, 0.9]])
    cmap = matplotlib.colors.ListedColormap(colors)

    # Initialize the activation and influence (input*weight)
    activation = []
    inf_raw = np.zeros(data_set.shape)

    # Loop through the data set (only non-variant)
    for i in range(0, 24):

        # Get correct instance
        forpass = NeuralNet(model.fit.size).forward_pass(data_set[i],
                                                         weights=model.fit.weights,
                                                         biases=model.fit.biases)

        # Keep the activation and influence for that node
        activation.extend([forpass.fit.out0[node_num]])
        inf_raw[i] = forpass.fit.credit[node_num]

    # Normalize for color range
    credit = (inf_raw - inf_raw.min()) / (inf_raw.max() - inf_raw.min())

    # Dimension assuming square
    img_size = np.sqrt(data_set[i].shape[0]).astype(int)

    # Reshape assumping square
    img = forpass.fit.weights[0][node_num][0:81].reshape(img_size, img_size)

    # Plot weights for the selected node
    imw = plt.imshow(img, cmap=plt.cm.coolwarm, alpha=.7, interpolation='bilinear')
    print(f"Weights for Node {node_num}")
    plt.show()

    # Loop through the data set (only non-variant)
    for i in range(0, 24):

        print(f"Activation = {activation[i]:.4f}")

        # Reshape assumping square
        img = data_set[i][0:81].reshape(img_size, img_size)

        # Show image
        plt.imshow(img, cmap=cmap)

        # Plot another layer showing infuence values
        influence = credit[i][0:81].reshape(img_size, img_size)

        im2 = plt.imshow(influence, vmin=0, vmax=1,
                         cmap=plt.cm.coolwarm, alpha=.5, interpolation='bilinear')
        plt.show()

def hidden_activations(data_set, letters, model):
    """Return hidden activations by node and letter as pandas df"""

    # Dictionary that will store dataframe input
    hidden_act = {}

    for i in range(0, data_set.shape[0]):

        # Get correct instance
        forpass = NeuralNet(model.fit.size).forward_pass(data_set[i],
                                                         weights=model.fit.weights,
                                                         biases=model.fit.biases)


        hidden_act[letters[i]+str(i)] = forpass.fit.out0.tolist()

    return pd.DataFrame.from_dict(hidden_act)

def display_all(case_input):
    """Plot all letters in a set"""

    for i in range(0, case_input.shape[0]):
        display_letter(case_input[i])

def main():
    """Load data, visualize, train model"""

    # Load data
    training_labels_1, training_labels_2, training_data, letters = process_data()

    # Define the gray box
    final_model_gray = NeuralNet((81, 3, 4))

    # Create random variants
    all_labels_1, all_labels_2, all_input, all_letters = randomize_input(training_labels_1,
                                                                         training_labels_2,
                                                                         training_data,
                                                                         letters,
                                                                         gamma=0.2,
                                                                         num_variants=10)

    # Visualize all letters (non-variant)
    display_all(training_data)

    # Train model for big classes
    final_model_gray.train(all_input, all_labels_2)

    for i in range(0, 3):
        display_node(training_data, final_model_gray, i)

    # Save table of hidden activations
    hidden_gray = hidden_activations(all_input, all_letters, final_model_gray)
    hidden_gray.to_csv('activations_gray.csv')

    gray_input = np.concatenate((all_input, hidden_gray.values.transpose()), axis=1)
    # Train model with gray box input
    final_model = NeuralNet((84, 5, 24)).train(gray_input, all_labels_1, epsilon=0.4)

    for i in range(0, 5):
        display_node(gray_input, final_model, i)

    # Save table of hidden activations
    hidden = hidden_activations(gray_input, all_letters, final_model)
    hidden.to_csv('activations.csv')

    return final_model

if __name__ == '__main__':
    FINAL_MODEL = main()
