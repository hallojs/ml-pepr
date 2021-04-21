import logging

import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

from pepr.privacy import gmia


def load_fashion_mnist():
    """Loads and preprocesses the MNIST dataset.

    Returns
    -------
    tuple
        (training data, training labels, test data, test labels)
    """
    train, test = tf.keras.datasets.fashion_mnist.load_data()
    train_data, train_labels = train
    test_data, test_labels = test

    # Normalize the data to a range between 0 and 1
    train_data = np.array(train_data, dtype=np.float32) / 255
    test_data = np.array(test_data, dtype=np.float32) / 255

    # Reshape the images to (28, 28, 1)
    train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
    test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

    train_labels = np.array(train_labels, dtype=np.int32)
    test_labels = np.array(test_labels, dtype=np.int32)

    return np.vstack((train_data, test_data)), np.hstack((train_labels, test_labels))


def create_model(input_shape, n_categories):
    """Architecture of the attacker and reference models.

    Parameters
    ----------
    input_shape : tuple
        Dimensions of the input for the target/training
    n_categories : int
        number of categories for the prediction
    models.

    Returns
    -------
    tensorflow.python.keras.engine.sequential.Sequential
        A convolutional neuronal network model.
    """
    model = Sequential()

    # first convolution layer
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding="same",
            input_shape=input_shape,
        )
    )
    model.add(Activation("relu"))

    # max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # second convolution layer
    model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding="same"))
    model.add(Activation("relu"))

    # max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"))

    # fully connected layer
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))

    # drop out
    model.add(Dropout(rate=0.5))

    # fully connected layer
    model.add(Dense(n_categories))
    model.add(Activation("softmax"))

    return model


def create_compile_model():
    input_shape = (28, 28, 1)
    number_classes = 10

    model = create_model(input_shape, number_classes)

    optimizer = optimizers.Adam(lr=0.0001)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]
    model.compile(optimizer, loss=loss, metrics=metrics)

    return model


def test_gmia_fashion_loaded():
    # Set up the logging system
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler("test.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger("pepr.privacy.gmia")
    logger.addHandler(file_handler)

    data, labels = load_fashion_mnist()

    target_model = models.load_model(
        "/Users/jonassander/Documents/Repositories/ml-pepr/tests/integration/fixtures/"
        "data_fashion_mnist/target_model"
    )
    target_models = target_model

    attack_pars = {
        "number_classes": 10,
        "number_reference_models": 100,
        "reference_training_set_size": 10000,
        "create_compile_model": create_compile_model,
        "reference_epochs": 50,
        "reference_batch_size": 50,
        "hlf_metric": "cosine",
        "hlf_layer_number": 10,
        "neighbor_threshold": 0.15,
        "probability_threshold": 0.1,
    }

    data_conf = {
        "reference_indices": list(range(40000)),
        "target_indices": list(range(40000, 50000)),
        "evaluation_indices": list(range(40000, 60000)),
        "record_indices_per_target": np.array([np.arange(10000)]),
    }

    reference_models_path = (
        "/Users/jonassander/Documents/Repositories/ml-pepr/tests/"
        "integration/fixtures/data_fashion_mnist/reference_model"
    )

    load_pars = {
        "records_per_reference_model": "/Users/jonassander/Documents/Repositories/"
        "ml-pepr/tests/integration/fixtures/data_fashion_mnist/"
        "records_per_reference_model.npy",
        "reference_models": [reference_models_path + str(i) for i in range(100)],
        "pairwise_distances_hlf_cosine": "/Users/jonassander/Documents/Repositories/"
        "ml-pepr/tests/integration/fixtures/data_fashion_mnist/"
        "pairwise_distances_hlf_cosine.npy",
    }

    name = "GMIA Test - Loaded Pars - Single Target"
    gmia_attack = gmia.DirectGmia(
        name, attack_pars, data, labels, data_conf, target_models
    )
    gmia_attack.run(load_pars=load_pars)

    # Check whether the attack precision for the cut-off-p-values 0.01, ..., 0.05 is
    # greater 0.5 and the recall for at least one cut-off-p-value is greater than 0.
    results = gmia_attack.attack_results
    precision = np.array(results["overall_precision"])[1:6]
    recall = np.array(results["overall_recall"])[1:6]
    assert np.sum(precision > 0.5) == 5
    assert np.sum(recall) > 0
