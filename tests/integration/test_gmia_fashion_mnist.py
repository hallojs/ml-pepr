import logging

import numpy as np
import tensorflow as tf

from pepr.privacy.gmia import DirectMia as gmia


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

    # Convert a class vector (integers) to binary class matrix (one-hot enc.)
    train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=10)
    test_labels_cat = tf.keras.utils.to_categorical(test_labels, num_classes=10)

    return (
        train_data,
        train_labels,
        train_labels_cat,
        test_data,
        test_labels,
        test_labels_cat,
    )


def test_gmia_fashion_loaded():
    # Set up the logging system
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('test.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger('pepr.gmia')
    logger.addHandler(file_handler)

    (
        train_data,
        train_labels,
        train_labels_cat,
        test_data,
        test_labels,
        test_labels_cat,
    ) = load_fashion_mnist()

    np.random.seed(0)
    np.random.shuffle(train_data)
    np.random.seed(0)
    np.random.shuffle(train_labels)
    np.random.seed(0)
    np.random.shuffle(train_labels_cat)

    np.random.seed(0)
    np.random.shuffle(test_data)
    np.random.seed(0)
    np.random.shuffle(test_labels)
    np.random.seed(0)
    np.random.shuffle(test_labels_cat)

    # Used to train the reference models
    reference_train_data = train_data[:40000]
    reference_train_labels = train_labels[:40000]
    reference_train_labels_cat = train_labels_cat[:40000]

    # Used to train the target models
    target_train_data = train_data[40000:50000]
    target_train_labels = train_labels[40000:50000]
    target_train_labels_cat = train_labels_cat[40000:50000]

    # Used for the evaluation of the attack
    attack_eval_data = train_data[40000:]
    attack_eval_labels = train_labels[40000:]
    attack_eval_labels_cat = train_labels_cat[40000:]

    input_shape = (28, 28, 1)
    number_classes = 10

    # Loading training datasets for reference models
    background_knowledge_size = 40000
    training_set_size = 10000
    path = "fixtures/data_fashion_mnist/records_per_reference_model.npy"
    records_per_reference_model = np.load(path)

    # Load target model
    path = "fixtures/data_fashion_mnist/target_model"
    epochs = 50
    batch_size = 50
    target_model = tf.keras.models.load_model(path)

    # Load reference models
    number_reference_models = 100
    save_path = "fixtures/data_fashion_mnist/reference_model"
    reference_models = gmia.load_models(save_path, number_reference_models)

    # Load high-level-features
    path = "fixtures/data_fashion_mnist/reference_high_level_features.npy"
    reference_high_level_features = np.load(path)

    path = "fixtures/data_fashion_mnist/target_high_level_features.npy"
    target_high_level_features = np.load(path)

    # Load cosine distances
    path = (
        "fixtures/data_fashion_mnist/pairwise_distances_high_level_features_cosine.npy"
    )
    distances = np.load(path)

    # Determine Target Records
    neighbor_threshold = 0.15
    probability_threshold = 0.1
    background_knowledge_size = 40000
    training_set_size = 10000
    print_target_statistics = True

    target_records = gmia.select_target_records(
        neighbor_threshold,
        probability_threshold,
        background_knowledge_size,
        training_set_size,
        distances
    )

    # Direct inference attack
    reference_inferences = gmia.get_model_inference(
        target_records, attack_eval_data, attack_eval_labels_cat, reference_models
    )

    target_inferences = gmia.get_model_inference(
        target_records, attack_eval_data, attack_eval_labels_cat, target_model
    )

    (
        used_target_records,
        pchip_references,
        ecdf_references,
    ) = gmia.sample_reference_losses(
        target_records, reference_inferences, print_target_information=True
    )

    cut_off_p_value = 0.05

    # The first 10000 samples in the evaluation data were used in the training of
    # the target model
    ground_truth = np.arange(0, 10000)

    members, _, _ = gmia.hypothesis_test(
        cut_off_p_value, pchip_references, target_inferences, used_target_records
    )

    true_positives = np.count_nonzero(np.less(members, len(ground_truth)))
    false_positives = len(members) - true_positives
    assert true_positives > false_positives
