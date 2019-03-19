'''
Simple CNN with Layer Visualization - Alan Kessler

Trains a convolutional neural network on MNIST

Based on these examples:

    - https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
    - https://github.com/yashk2810/Visualization-of-Convolutional-Layers/blob/master/Visualizing%20Filters%20Python3%20Theano%20Backend.ipynb
    - https://github.com/slundberg/shap

Trained using a p2.xlarge instance on AWS with:
Deep Learning AMI (Ubuntu) Version 16.0 - ami-0e0e0d5bbfb3508be
TensorFlow backend
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.core import Activation
import matplotlib.pyplot as plt
import shap
import numpy as np

# %matplotlib inline

class ConvModel:
    """Defines a simple convolutional neural net for MNIST"""

    def __init__(self, num_classes=10, input_shape=(28, 28, 1)):
        """Defines MNIST inputs and outputs"""
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        (self.x_train, self.y_train) = (None, None)
        (self.x_test, self.y_test) = (None, None)

    def load_data(self):
        """Load and format MNIST data"""

        print("Loading Data")

        # Load data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Reshape X for use in TensorFlow
        x_train = x_train.reshape(x_train.shape[0], self.input_shape[0],
                                  self.input_shape[1], self.input_shape[2])
        x_test = x_test.reshape(x_test.shape[0], self.input_shape[0],
                                self.input_shape[1], self.input_shape[2])

        print(f"{self.input_shape[0]}x{self.input_shape[1]} image size")

        # Define the inputs as float
        self.x_train = x_train.astype('float32')
        self.x_test = x_test.astype('float32')

        # Normalize the inputs (MNIST in range (0, 255))
        self.x_train /= 255
        self.x_test /= 255

        # One hot encode labels
        self.y_train = keras.utils.to_categorical(y_train, self.num_classes)
        self.y_test = keras.utils.to_categorical(y_test, self.num_classes)

        return self

    def spec_model(self):
        """Specify CNN"""

        print("Specifying Model")

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              input_shape=self.input_shape,
                              activation='relu',
                              name='conv1'))

        self.model.add(Conv2D(64, (3, 3),
                              activation='relu',
                              name='conv2'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool'))
        self.model.add(Dropout(0.25, name='dropout1'))
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(128, activation='relu', name='dense1'))
        self.model.add(Dropout(0.5, name='dropout2'))
        self.model.add(Dense(self.num_classes, activation='softmax', name='dense2'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

        return self

    def spec_model1(self):
        """Specify CNN - without dropout"""

        print("Specifying Model")

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              input_shape=self.input_shape,
                              activation='relu',
                              name='conv1'))

        self.model.add(Conv2D(64, (3, 3),
                              activation='relu',
                              name='conv2'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool'))
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(128, activation='relu', name='dense1'))
        self.model.add(Dense(self.num_classes, activation='softmax', name='dense2'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

        return self

    def spec_model2(self):
        """Specify CNN - without dropout, conv2"""

        print("Specifying Model")

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              input_shape=self.input_shape,
                              activation='relu',
                              name='conv1'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool'))
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(128, activation='relu', name='dense1'))
        self.model.add(Dense(self.num_classes, activation='softmax', name='dense2'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])


        return self

    def spec_model3(self):
        """Specify CNN - without dropout, conv2, half dense"""

        print("Specifying Model")

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              input_shape=self.input_shape,
                              activation='relu',
                              name='conv1'))

        self.model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool'))
        self.model.add(Flatten(name='flatten'))
        self.model.add(Dense(64, activation='relu', name='dense1'))
        self.model.add(Dense(self.num_classes, activation='softmax', name='dense2'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])


        return self

    def train(self, batch_size=128, epochs=12):
        """Train model"""

        if self.x_train is None:
            self.load_data()

        if self.model is None:
            self.spec_model()

        self.model.fit(self.x_train, self.y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       validation_data=(self.x_test, self.y_test))

        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nTest loss: {score[0]:.4f}")
        print(f"Test accuracy: {score[1]:.4f}")

        return self

    def visualize(self, layer, image_number):
        """Visualize layer output"""

        # Define a function that gives layer output for a given input
        inputs = [K.learning_phase()] + self.model.inputs
        _layer_outputs = K.function(inputs, [layer.output])

        # Format data to see layer outputs
        def layer_outputs(image_data):
            """Removes the training phase flag"""
            return _layer_outputs([0] + [image_data])

        image = np.expand_dims(self.x_train[image_number], axis=0)
        layer_data = layer_outputs(image)
        layer_data = np.squeeze(layer_data)

        # Define grid shape of plots
        n = layer_data.shape[2]
        n = int(np.ceil(np.sqrt(n)))

        # Visualization of each filter of the layer
        fig = plt.figure(figsize=(12, 8))
        for i in range(layer_data.shape[2]):
            ax = fig.add_subplot(n, n, i+1)
            ax.imshow(layer_data[:, :, i], cmap='gray')

        plt.show()

def shap_vis(obj):
    """Generate shap visuals for a given model object"""

    training = obj.x_train
    testing = obj.x_test

    # Select a set of background examples to take an expectation over
    background = training[np.random.choice(training.shape[0], 1000, replace=False)]

    # Explain predictions of the model on fixed image indexes from test
    e = shap.DeepExplainer(obj.model, background)
    shap_values = e.shap_values(testing[[12, 41, 1, 2]])

    # Plot the feature attributions
    shap.image_plot(shap_values, -testing[[12, 41, 1, 2]])



if __name__ == '__main__':

    def main(images=[12, 41]):
        """Run the analysis - Fit CNN and Visualize for an image"""

        # Initialize and load data
        simple_cnn = ConvModel()
        simple_cnn.load_data()

        # Display model structure
        simple_cnn.spec_model()
        simple_cnn.model.summary()

        # Train the full model
        simple_cnn.train()

        # Visualize the filters generated by the layers
        for i in images:
            # Display the image
            plt.imshow(simple_cnn.x_train[i][:, :, 0], cmap='gray')
            plt.show()

            # Display the first conv layer
            print("Convolutional Layer 1")
            simple_cnn.visualize(simple_cnn.model.get_layer('conv1'), i)

            # Display the second conv layer
            print("Convolutional Layer 2")
            simple_cnn.visualize(simple_cnn.model.get_layer('conv2'), i)

        # Visualize feature attribution
        shap_vis(simple_cnn)

        # Train model without dropout
        simple_cnn.spec_model1()
        simple_cnn.model.summary()
        simple_cnn.train()
        shap_vis(simple_cnn)

        # Train model without dropout
        simple_cnn.spec_model2()
        simple_cnn.model.summary()
        simple_cnn.train()
        shap_vis(simple_cnn)

        # Train model without dropout
        simple_cnn.spec_model3()
        simple_cnn.model.summary()
        simple_cnn.train()
        shap_vis(simple_cnn)

        return simple_cnn

    main()
