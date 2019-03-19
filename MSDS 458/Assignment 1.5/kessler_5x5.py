# -*- coding: utf-8 -*-
"""Training a simple Neural Network using numpy - Alan Kessler

Assumptions:
    * Assumes sigmoid transfer function
    * Assumes a single layer of hidden nodes
    * Assumes batch size is fixed at 1

"""

import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors


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

        # Store results
        forpass = collections.namedtuple('ForwardPass',
                                         ['case_input', 'size', 'alpha',
                                          'weights', 'biases', 'net0', 'out0',
                                          'net1', 'out1'])

        self.fit = forpass(case_input, self.size, self.alpha, weights, biases,
                           net0, out0, net1, out1)

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

    def train(self, data_set, target, eta=0.5, epsilon=0.05, iterations=5000, verbose=True):
        """Trains the neural network and predicts target

        Assumptions:
            Each iteration (epoch) uses all input records in random order

        Args:
            data_set: array of input data (multiple observations)
            target: array of target output (multiple observations)
            eta: learning rate (default 0.5)
            epsilon: early stopping criteria SSE (default 0.05)
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

        if verbose:
            print("Training Model")
            print("\nInput:")
            print(data_set)
            print("\nTarget:")
            print(target)
            print("\n Total SSE by Iteration")
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
            if verbose and iteration_num % 100 == 0:
                print(f"{iteration_num:4}. {sse:.4f}")

            # Enable early stopping past the threshold set
            if sse <= epsilon or iteration_num == iterations:
                if verbose:
                    for i in range(0, data_set.shape[0]):
                        data_single = data_set[i]
                        target_single = target[i]
                        self.forward_pass(data_single, weights=self.fit.weights,
                                          biases=self.fit.biases)
                        print("\n------------")
                        print("\nInput array:")
                        print(self.fit.case_input)
                        print("\nTarget array")
                        print(target_single)
                        print("\nHidden Layer Output:")
                        print(self.fit.out0)
                        print("\nOutput Layer Final Output:")
                        print(self.fit.out1)
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

def display_all(case_input):
    """Plot all letters in a set"""

    for i in range(0, case_input.shape[0]):
        display_letter(case_input[i])

def main():
    """Load data, visualize, train model"""

    # Load data
    training_data = np.array([[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
                               1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0,
                               1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
                               1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                              [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1,
                               1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                              [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1,
                               1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]])
    training_labels = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])

    # Visualize letters
    display_all(training_data)

    # Train model
    final_model = NeuralNet((25, 6, 5)).train(training_data, training_labels)

    return final_model

if __name__ == '__main__':
    main()
