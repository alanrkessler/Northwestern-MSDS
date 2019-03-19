# -*- coding: utf-8 -*-
"""Training XOR Model using numpy - Alan Kessler

Assumptions:
    * Assumes sigmoid transfer function with alpha=1
    * Assumes a single layer of hidden nodes
    * Assumes batch size is fixed at 1

"""

import numpy as np

# Random number generating seed for reproducible results
np.random.seed(9798)

class NeuralNet:
    """Describes a neural network of the type used in this example"""
    def __init__(self, case_input, weights, biases, net0, out0, net1, out1):
        self.case_input = case_input
        self.weights = weights
        self.biases = biases
        self.net0 = net0
        self.out0 = out0
        self.net1 = net1
        self.out1 = out1

    def describe(self):
        """Prints information about the neural network"""
        print("Input array:")
        print(self.case_input)
        print("\nWeights (first row corresponds to first output):")
        print(self.weights)
        print("\nBiases:")
        print(self.biases)
        print("\nHidden Layer Net:")
        print(self.net0)
        print("\nHidden Layer Output:")
        print(self.out0)
        print("\nOutput Layer Net:")
        print(self.net1)
        print("\nOutput Layer Final Output:")
        print(self.out1)

class Backprop:
    """Describes the results of backpropagation"""
    def __init__(self, target, eta, delta0, delta1, weights_new, biases_new):
        self.target = target
        self.eta = eta
        self.delta0 = delta0
        self.delta1 = delta1
        self.weights_new = weights_new
        self.biases_new = biases_new
    def describe(self):
        """Prints information about the backpropagation"""
        print("Target:")
        print(self.target)
        print(f"\nEta (learning rate): {self.eta}")
        print("\nDelta 0 (closest to output):")
        print(self.delta0)
        print("\nDelta 1 (closest to input):")
        print(self.delta1)
        print("\nNew Weights (first row corresponds to first output):")
        print(self.weights_new)
        print("\n Biases (unchanged):")
        print(self.biases_new)

def sigmoid_transfer(net_sum):
    """Calculate the sigmoid transfer function.

    Args:
        net_sum: the input into the node.
    Returns:
        the activation output of the node.
    Raises:
        TypeError: if net_sum is not a number.

    """
    return 1 / (1 + np.exp(-net_sum))

def initial_weights(input_dim, hidden_nodes, output_dim):
    """Randomly assigns initial weights [-1, 1)

    Args:
        input_dim: number of inputs
        hidden_nodes: number of hidden_nodes
        output_dim: number of output classes
    Returns:
        list of initial weight arrays

    """
    input_to_hidden = np.random.uniform(-1, 1, (hidden_nodes, input_dim))
    hidden_to_output = np.random.uniform(-1, 1, (output_dim, hidden_nodes))
    return [input_to_hidden, hidden_to_output]

def initial_biases(hidden_nodes, output_dim):
    """Randomly assigns initial biases [-1, 1)

    Args:
        hidden_nodes: number of hidden_nodes
        output_dim: number of output classes
    Returns:
        list of initial weight arrays

    """
    input_to_hidden = np.random.uniform(-1, 1, hidden_nodes)
    hidden_to_output = np.random.uniform(-1, 1, output_dim)
    return [input_to_hidden, hidden_to_output]

def forward_pass(case_input, size,
                 weights=None,
                 biases=None,
                 verbose=False):
    """Calculate the result of a forward pass through the neural network

    Args:
        case_input: array of input into the neural network
        size: tuple of input size, hidden nodes, and output classes
        weights: array of network weights (default None)
        biases: array of network biases (default None)
        verbose: if true, prints full results (default False)
    Returns:
        a neural network (NeuralNet object) containing results
    Raises:
        TypeError: if arguments are not arrays.

    """

    # If not supplied, generate random weights
    if weights is None:
        weights = initial_weights(size[0], size[1], size[2])
    if biases is None:
        biases = initial_biases(size[1], size[2])

    # Check input types
    if not isinstance(case_input, np.ndarray):
        raise TypeError("case_input must be an array")
    if not isinstance(size, tuple):
        raise TypeError("case_input must be an array")
    if not isinstance(weights[0], np.ndarray):
        raise TypeError("weights must be an array")
    if not isinstance(biases[0], np.ndarray):
        raise TypeError("biases must be an array")

    # Calculate the summed input to the hidden layer
    net0 = np.matmul(case_input, np.transpose(weights[0])) + biases[0]

    # Apply sigmoid transfer function
    out0 = sigmoid_transfer(net0)

    # Calculate the summed input to the output layer
    net1 = np.matmul(out0, np.transpose(weights[1])) + biases[1]

    # Apply sigmoid transfer function
    out1 = sigmoid_transfer(net1)

    forpass = NeuralNet(case_input, weights, biases, net0, out0, net1, out1)

    if verbose:
        forpass.describe()

    return forpass

def sse_eval(target, network, verbose=False):
    """Calculate the sum of squared errors for a neural network

    Args:
        target: array of ideal output
        network: NeuralNet object
        verbose: if true, print result (default False)
    Returns:
        sum of squared errors from the network
    Raises:
        TypeError: if arguments are not an array and a NeuralNet.

    """

    # Check input types
    if not isinstance(target, np.ndarray):
        raise TypeError("target must be an array")
    if not isinstance(network, NeuralNet):
        raise TypeError("network must be a NeuralNet")

    sse = np.sum((target - network.out1)**2)

    if verbose:
        print(f"Total SSE = {sse:.4f}")

    return sse

def backpropagation(target, network, eta, verbose=False):
    """Conducts a step of backpropagation

    Args:
        target: array of ideal output
        network: NeuralNet object
        eta: learning rate
        verbose: if true, print result of backpropagation (default False)
    Returns:
        results of backpropagation (a Backprop object)
    Raises:
        TypeError: if arguments are not an array, NeuralNet, and numeric.

    """

    # Check if eta is a number
    if not isinstance(eta, (int, float)):
        raise TypeError("eta (learning rate) must be type int or float")

    # Working backwards so delta0 is array of deltas closest to output layer
    delta0 = -1*(target - network.out1)*network.out1*(1 - network.out1)

    # Adjusted weights from hidden to output layer
    weights0 = network.weights[1] - eta*np.outer(delta0, network.out0)

    # Adjust bias from hidden to output layer
    biases0 = network.biases[1] - eta*delta0

    # Working backwards so delta1 is array of deltas closest to input layer
    summation = np.sum(np.transpose(network.weights[1])*delta0, axis=1)
    delta1 = summation*network.out0*(1 - network.out0)

    # Adjusted weights from input to hidden layer
    weights1 = network.weights[0] - eta*np.outer(delta1, network.case_input)

    # Adjust bias from input to hidden layer
    biases1 = network.biases[0] - eta*delta1

    # Create a new weights array containing all weights
    weights_new = [weights1, weights0]

    # Create a new biases array containing all biases
    biases_new = [biases1, biases0]

    bprop = Backprop(target, eta, delta0, delta1, weights_new, biases_new)

    if verbose:
        bprop.describe()

    return bprop

def train_xor(case_input, target, size,
              eta=0.5, epsilon=0.05, iterations=5000,
              verbose=True):
    """Trains the XOR model and then predicts target

    Assumptions:
        For each iteration (epoch) all input records are used in random order

    Args:
        case_input: array of input data (four observations)
        target: array of target output (four observations)
        size: tuple of input size, hidden nodes, and output classes
        eta: learning rate (default 0.5)
        epsilon: early stopping criteria SSE (default 0.05)
        iterations: number of maximum training iterations (default 5000)
        verbose: if true, print results (default True)
    Returns:
        a neural network object (NeuralNet) containing trained results
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

    # Calculate a baseline SSE
    sse = 0
    for i in range(0, case_input.shape[0]):
        case_input_single = case_input[i]
        target_single = target[i]
        fit = forward_pass(case_input_single, size)
        sse += sse_eval(target_single, fit)

    if verbose:
        print("Training Model")
        print("\nInput:")
        print(case_input)
        print("\nTarget:")
        print(target)
        print("\n Total SSE by Iteration")
        print(f"\n{0:4}. {sse:.4f}")

    # Train over the specified number of iterations
    for iteration_num in range(1, iterations+1):
        # Perform backpropagation for each training record
        shuffle = np.random.choice(case_input.shape[0],
                                   case_input.shape[0],
                                   replace=False)
        for i in shuffle:
            case_input_single = case_input[i]
            target_single = target[i]
            fit = forward_pass(case_input_single, size,
                               weights=fit.weights,
                               biases=fit.biases)
            backprop = backpropagation(target_single, fit, eta)
            fit = forward_pass(case_input_single, size,
                               weights=backprop.weights_new,
                               biases=backprop.biases_new)
        # Calculate total SSE for the iteration
        sse = 0
        for i in range(0, case_input.shape[0]):
            case_input_single = case_input[i]
            target_single = target[i]
            fit = forward_pass(case_input_single, size,
                               weights=backprop.weights_new,
                               biases=backprop.biases_new)
            sse += sse_eval(target_single, fit)
        if verbose and iteration_num % 100 == 0:
            print(f"{iteration_num:4}. {sse:.4f}")

        # Enable early stopping past the threshold set
        if sse <= epsilon or iteration_num == iterations:
            if verbose:
                for i in range(0, case_input.shape[0]):
                    case_input_single = case_input[i]
                    target_single = target[i]
                    fit = forward_pass(case_input_single, size,
                                       weights=backprop.weights_new,
                                       biases=backprop.biases_new)
                    print("\n------------")
                    print("\nInput array:")
                    print(fit.case_input)
                    print("\nTarget array")
                    print(target_single)
                    print("\nWeights (first row corresponds to first output):")
                    print(fit.weights)
                    print("\nBiases:")
                    print(fit.biases)
                    print("\nOutput Layer Final Output:")
                    print(fit.out1)
            break

    return fit

if __name__ == '__main__':
    train_xor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
              np.array([[0, 1], [1, 0], [1, 0], [0, 1]]),
              (2, 2, 2))
