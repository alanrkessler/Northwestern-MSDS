'''
X-Ray Classification of Pneumonia - Alan Kessler

Trains a convolutional neural network on Pneumonia data

Data was downloaded to the instance from:

    - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Trained using a p2.xlarge instance on AWS with:
Deep Learning AMI (Ubuntu) Version 16.0 - ami-0e0e0d5bbfb3508be
Keras with TensorFlow backend
Self-contained analysis/tutorial, so this does not adhere to PEP-8
'''


import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import set_random_seed
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras import backend
import shap
from vis.utils import utils

# Set seeds so that results are more reproducible
# Certain random initializations still may not exactly match
np.random.seed(438)
tensorflow.set_random_seed(754)

# Data Exploration

# Display example data for each class
# Data directory structure modified to remove redundant layers
image_normal = load_img('./chest_xray/train/NORMAL/IM-0115-0001.jpeg')
image_pneu = load_img('./chest_xray/train/PNEUMONIA/person2_bacteria_3.jpeg')

# Plot the examples side by side for comparison
figure, (axis1, axis2) = plt.subplots(1, 2)
axis1.imshow(image_normal)
axis1.set_title('Normal')
axis2.imshow(image_pneu)
axis2.set_title('Pneumonia')
figure.tight_layout()
plt.show()

# Define data for use by Keras

# Set high-level parameters
# Selected for reasonable run times
image_height, image_width = 128, 128
batch_size = 32

# Specify training/validation structure
# Rescale and randomize to improve generalization
# Parameters select from Keras documentation examples
# Kaggle Validation set will be used for visuals due to small size
# Create new validtion partition from the training data
gen_train = ImageDataGenerator(rescale=1/255,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=True,
                               validation_split=0.2)

batches_train = gen_train.flow_from_directory('./chest_xray/train',
                                              target_size=(image_height, image_width),
                                              color_mode='grayscale',
                                              seed=9798,
                                              batch_size=batch_size,
                                              subset='training')

batches_val = gen_train.flow_from_directory('./chest_xray/train',
                                            target_size=(image_height, image_width),
                                            color_mode='grayscale',
                                            seed=5748,
                                            batch_size=batch_size,
                                            subset='validation')

# Specify test/interpretation structure
# Test images are scaled by not randomly adjusted
# Random adjustments would not match how model would be used

# Shuffle is false and batch size is 1
# This avoids getting different fit statistics each run
gen_test = ImageDataGenerator(rescale=1/255)

batches_test = gen_test.flow_from_directory('./chest_xray/test',
                                            target_size=(image_height, image_width),
                                            color_mode='grayscale',
                                            seed=6170,
                                            shuffle=False,
                                            batch_size=1)

batches_int = gen_test.flow_from_directory('./chest_xray/val',
                                           target_size=(image_height, image_width),
                                           color_mode='grayscale',
                                           seed=2281,
                                           shuffle=False,
                                           batch_size=1)

# Specify the models
# Architecture based on VGG-16
# Each mdoel is more complex (more layers) than the last
def spec_model_1():
    """Specify first model"""

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=(image_height, image_width, 1),
                     activation='relu', padding='same',
                     name='conv_1_1'))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     name='conv_1_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(32, activation='relu', name='fc_1'))
    model.add(Dropout(0.5, name='dropout_1'))
    model.add(Dense(2, activation='sigmoid', name='fc_2'))

    # Define optimizer & compile model
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=rms,
                  metrics=['accuracy'])

    return model

def spec_model_2():
    """Specify second model"""

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=(image_height, image_width, 1),
                     activation='relu', padding='same',
                     name='conv_1_1'))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     name='conv_1_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1'))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     input_shape=(image_height, image_width, 1),
                     activation='relu', padding='same',
                     name='conv_2_1'))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     name='conv_2_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(64, activation='relu', name='fc_1_1'))
    model.add(Dense(64, activation='relu', name='fc_1_2'))
    model.add(Dropout(0.5, name='dropout_1'))
    model.add(Dense(2, activation='sigmoid', name='fc_2'))

    # Define optimizer & compile model
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=rms,
                  metrics=['accuracy'])

    return model

def spec_model_3():
    """Specify third model"""

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     input_shape=(image_height, image_width, 1),
                     activation='relu', padding='same',
                     name='conv_1_1'))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     name='conv_1_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1'))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     input_shape=(image_height, image_width, 1),
                     activation='relu', padding='same',
                     name='conv_2_1'))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     name='conv_2_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_2'))
    model.add(Conv2D(128, kernel_size=(3, 3),
                     input_shape=(image_height, image_width, 1),
                     activation='relu', padding='same',
                     name='conv_3_1'))
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu', padding='same',
                     name='conv_3_2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_3'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(128, activation='relu', name='fc_1_1'))
    model.add(Dense(128, activation='relu', name='fc_1_2'))
    model.add(Dropout(0.5, name='dropout_1'))
    model.add(Dense(2, activation='sigmoid', name='fc_2'))

    # Define optimizer & compile model
    rms = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=rms,
                  metrics=['accuracy'])

    return model

# Add early stopping based on validation
# Will avoid excess epochs and generalize better
early_stop = EarlyStopping(monitor='val_loss', patience=3)


def plot_hist(fit_hist):
    """Plot training history by epoch for binary response.

    Args:
        fit_hist: training history.
    Returns:
        Plots of the .
    Raises:
        TypeError: if fit_hist is not a keras history.
    """

    if not isinstance(fit_hist, keras.callbacks.History):
        raise TypeError(f"{fit_hist} must be an array")

    # Count the number of epochs
    epochs = len(fit_hist.history['loss'])

    # Plot loss by epoch
    plt.title('Loss by Epoch')
    plt.xticks(np.arange(0, epochs+1, 1.0))
    plt.plot(fit_hist.history['loss'])
    plt.plot(fit_hist.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'],
               loc='upper right')
    plt.show()

    # Plot accuracy by epoch
    plt.title('Accuracy by Epoch')
    plt.xticks(np.arange(0, epochs+1, 1.0))
    plt.plot(fit_hist.history['acc'])
    plt.plot(fit_hist.history['val_acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'],
               loc='upper left')
    plt.show()


def model_stats(model_name, testing=batches_test):
    """Displays M

    Args:
        model: string of the saved model name.
        testing:
    Returns:
        Plots of the .

    """

    # Reset the test batches so they line up with labels
    batches_test.reset()

    # Calculate test statistics
    model = load_model(f"{model_name}.h5")
    test_res = model.evaluate_generator(testing,
                                        steps=len(testing))

    # Get the test labels
    test_labels = []

    for i in range(0, len(batches_test)):
        test_labels.extend(batches_test[i][1][:,1])

    test_labels = np.array(test_labels)

    # Get the predictions
    y_pred = model.predict_generator(batches_test,
                                     steps=len(batches_test))

    # Calculate AUC
    fpr, tpr, thresholds = roc_curve(test_labels, y_pred[:, 1])
    test_auc = auc(fpr, tpr)

    # Plot the ROC curve with AUC label
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    model_label = f"{model_name.replace('_',' ').title()}"
    model_perf = f"Loss = {test_res[0]:.4f}, Accuracy: {test_res[1]*100:.2f}%"
    plt.plot(fpr, tpr, label=f"{model_label} (AUC = {test_auc:.3f})\n{model_perf}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{model_label} - ROC Curve for Test Data")
    plt.legend(loc='best')
    plt.show()

    # Calculate lift on the test
    lift = pd.DataFrame(data={'pred': y_pred[:,1], 'label': test_labels})
    lift['decile'] = pd.qcut(lift['pred'], 5, labels=False, duplicates='drop')
    lift_agg = lift.groupby('decile').agg(['mean'])
    lift_agg.columns = lift_agg.columns.droplevel(level=1)

    # Plot the lift
    lift_chart = lift_agg[['label']].plot(kind='bar',
                                          title=f"{model_label} - Lift for Test Data",
                                          legend=False)
    lift_chart.set_xlabel("Prediction Quantile")
    lift_chart.set_ylabel("Average Pneumonia Frequency")
    lift_chart.set_xticklabels(lift_chart.xaxis.get_majorticklabels(),
                               rotation=0)
    figure.tight_layout()
    plt.show()


# Specify the model
model_1 = spec_model_1()

# Display model structure
model_1.summary()

# Save the early stopping model as the final model
# Otherwise the model that did not improved would be saved
model_1_checkpoint = ModelCheckpoint('model_1.h5', monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')

# Train the model
# Epoch number chosen to give early stopping room
history_1 = model_1.fit_generator(batches_train,
                                  steps_per_epoch=len(batches_train),
                                  validation_data=batches_val,
                                  validation_steps=len(batches_val),
                                  epochs=15,
                                  callbacks=[early_stop, model_1_checkpoint])

# Plot training history
plot_hist(history_1)

# Plot model statistics
model_stats('model_1')

# Specify the model
model_2 = spec_model_2()

# Display model structure
model_2.summary()

# Save the early stopping model as the final model
# Otherwise the model that did not improved would be saved
model_2_checkpoint = ModelCheckpoint('model_2.h5', monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')

# Train the model
# Epoch number chosen to give early stopping room
history_2 = model_2.fit_generator(batches_train,
                                  steps_per_epoch=len(batches_train),
                                  validation_data=batches_val,
                                  validation_steps=len(batches_val),
                                  epochs=15,
                                  callbacks=[early_stop, model_2_checkpoint])

# Plot training history
plot_hist(history_2)

# Plot model statistics
model_stats('model_2')

# Specify the model
model_3 = spec_model_3()

# Display model structure
model_3.summary()

# Save the early stopping model as the final model
# Otherwise the model that did not improved would be saved
model_3_checkpoint = ModelCheckpoint('model_3.h5', monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')

# Train the model
# Epoch number chosen to give early stopping room
history_3 = model_3.fit_generator(batches_train,
                                  steps_per_epoch=len(batches_train),
                                  validation_data=batches_val,
                                  validation_steps=len(batches_val),
                                  epochs=15,
                                  callbacks=[early_stop, model_3_checkpoint])

# Plot training history
plot_hist(history_3)

# Plot model statistics
model_stats('model_3')

# Use Shap to visualize credit assignment for Model 1

# Reset the generator to allow for reproducible sets
batches_val.reset()
batches_train.reset()

# Select a set of background examples to take an expectation over
# The entire set is not used due to memory constraints
background = batches_train[0][0]
for i in range(1, 6):
    background = np.concatenate((background, batches_train[i][0]), axis=0)

# Load the model to take expectation
model = load_model('model_1.h5')
model_explain = shap.DeepExplainer(model, background)

# Generate shap values for validation data (0-Healthy, 1-Not Healthy)
shap_values_0 = model_explain.shap_values(batches_val[0][0][[2, 24]])
shap_values_1 = model_explain.shap_values(batches_val[0][0][[3, 5]])

# Verify true postives/negatives
print("Predictions for Validation Data")
print(model.predict(batches_val[0][0][[2, 24, 3, 5]]))

# Plot feature attributions
print("\nHealthy - True Negatives")
shap.image_plot(shap_values_0[1], -batches_val[0][0][[2, 24]])
print("Pneumonia - True Positives")
shap.image_plot(shap_values_1[1], -batches_val[0][0][[3, 5]])


def visualize(layer_name):
    """Visualize layer output for Model 1"""

    # Define a function that gives layer output for a given input
    layer_idx = utils.find_layer_idx(model, layer_name)
    inputs = [backend.learning_phase()] + model.inputs
    _layer_outputs = backend.function(inputs, [model.layers[layer_idx].output])

    # Format data to see layer outputs
    def layer_outputs(image_data):
        """Removes the training phase flag"""
        return _layer_outputs([0] + [image_data])

    image = np.expand_dims(batches_val[0][0][5], axis=0)
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
    fig.suptitle(f"{layer_name} Features", fontsize=14)
    figure.tight_layout()

    plt.show()

visualize('conv_1_1')
visualize('conv_1_2')
