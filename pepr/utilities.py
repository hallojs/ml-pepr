"""Utility module containing utilities to speed up pentesting."""
import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def assign_record_ids_to_target_models(
    target_knowledge_size, number_target_models, target_training_set_size, offset=0
):
    """Create training datasets (index sets) for the target models.

    Each target training dataset contains exactly 50 percent of the given data. Each
    record is in the half of the data sets.

    Parameters
    ----------
    target_knowledge_size : int
        Number of data samples to be distributed to the target data sets.
    number_target_models : int
        Number of target models for which datasets should be created.
    target_training_set_size : int
        Number of records each target training set should be contain.
    offset : int
        If offset is zero, the lowest index in the resulting datasets is zero. If the
        offset is o, all indices i are shifted by o: i + o.
    Returns
    -------
    numpy.ndarray
        Index array that describes which record is used to train which model.
    """
    records_per_target_model = np.array([])
    for i in range(0, int(number_target_models / 2)):
        np.random.seed(i)
        selection = np.random.choice(
            np.arange(target_knowledge_size),
            target_knowledge_size,
            replace=False,
        )
        if i > 0:
            records_per_target_model = np.vstack(
                (records_per_target_model, selection[:target_training_set_size])
            )
            records_per_target_model = np.vstack(
                (records_per_target_model, selection[target_training_set_size:])
            )
        else:
            records_per_target_model = np.vstack(
                (
                    selection[:target_training_set_size],
                    selection[target_training_set_size:],
                )
            )

    return records_per_target_model + offset

def filter_out_outlier(data, labels, filter_pars, data_conf, save_path=None, load_pars=None):
    """
    Filter out potentially vulnerable samples.

    Steps:

    1. Create mapping of records to reference models.
    2. Train the reference models.
    3. Generate intermediate models.
    4. Extract reference high-level features.
    5. Extract target high-level features.
    6. Compute pairwise distances between reference and target high-level features.
    7. Determine potential vulnerable target records.
    8. Remove potential vulnerable target records.

    Parameters
    ----------
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.\
    filter_pars : dict
        Dictionary containing needed filter parameters:

        * number_reference_models (int): Number of reference models to be trained.
        * reference_training_set_size (int): Size of the trainings set for each
          reference model.
        * create_compile_model (function): Function that returns a compiled
          TensorFlow model (typically identical to the target model) used in the
          training of the reference models.
        * reference_epochs (int): Number of training epochs of the reference models.
        * reference_batch_size (int): Batch size used in the training of the
          reference models.
        * hlf_metric (str): Metric (typically 'cosine') used for the distance
          calculations in the high-level feature space. For valid metrics see
          documentation of sklearn.metrics.pairwise_distances.
        * hlf_layer_number (int): If value is n, the n-th layer of the model
          returned by create_compile_model is used to extract the high-level feature
          vectors.
        * neighbor_threshold (float): If distance is smaller then the neighbor
          threshold the record is selected as target record.
        * probability_threshold (float): For details see section 4.3 from the paper.
        * number_target_records (int): If set, the selection algorithm performs
          `max_search_rounds`, to find a `neighbor_threshold`, that leads to a finding
          of `n_targets` target records. These target records are most vulnerable with
          respect to our selection criterion.
        * max_search_rounds (int): If `number_target_records` is given, maximal
          `max_search_rounds` are performed to find `number_target_records` of potential
          vulnerable target records.

    data_conf : dict

        * reference_indices (list): List of indices describing which of the records
          from data are used to train the reference models.
        * target_indices (list): List of indices describing which of the records
          from data were used to train the target model(s).
        * evaluation_indices (list): List of indices describing which of the records
          from data are used to evaluate an attack. Typically these are to one half
          records used to train the target models and one half neither used to train
          the target model(s) or the reference models.

    save_path : str
        If path is given, the following (partly computational expensive)
        intermediate results are saved to disk:

        * The mapping of training-records to reference models
        * The trained reference models
        * The reference high-level features
        * The target high-level features
        * The matrix containing all pairwise distances between the reference- and
          target high-level features.

    load_pars : dict
        If this dictionary is given, the following computational intermediate
        results can be loaded from disk.

        * records_per_reference_model (str) : Path to the mapping.
        * reference_models (list) : List of paths to the reference models.
        * pairwise_distance_hlf_<hlf_metric> (str) :  Path to the pairwise distance
          matrix between the reference- and target high-level features using a
          hlf_metric (e.g. cosine).

    Returns
    -------
    numpy.ndarray:
        Target indices without outliers.
    float:
        Calculated neighbor threshold of the result.
    float:
        Calculated probability threshold of the result.
    """
    from pepr.privacy.gmia import DirectGmia
    from tensorflow.keras import models, utils

    labels_cat = utils.to_categorical(
        labels, num_classes=filter_pars["number_classes"]
    )

    # Slice data set
    # -- Used to train the reference models
    reference_train_data = data[data_conf["reference_indices"]]
    reference_train_labels_cat = labels_cat[
        data_conf["reference_indices"]
    ]

    # -- Used to train the target models
    target_train_data = data[data_conf["target_indices"]]

    # -- Used for the evaluation of an attack
    attack_eval_data = data[data_conf["evaluation_indices"]]

    load = load_pars is None

    # Step 1
    if load or "records_per_reference_model" not in load_pars.keys():
        # -- Compute Step 1
        logger.info("Create mapping of records to reference models.")
        records_per_reference_model = (
            DirectGmia._assign_records_to_reference_models(
                filter_pars["number_reference_models"],
                len(reference_train_data),
                filter_pars["reference_training_set_size"],
            )
        )
        # -- Save Step 1
        if save_path is not None:
            path = save_path + "/records_per_reference_model.npy"
            logger.info(f"Save mapping of records to reference models: {path}.")
            np.save(path, records_per_reference_model)
    else:
        # -- Or Load Step 1
        path = load_pars["records_per_reference_model"]
        logger.info(f"Load mapping of records to reference models: {path}.")
        records_per_reference_model = np.load(path)

    logger.debug(
        f"records_per_reference_model shape: "
        f"{records_per_reference_model.shape}"
    )

    # Step 2
    if load or "reference_models" not in load_pars.keys():
        # -- Compute Step 2
        reference_models = DirectGmia._train_reference_models(
            filter_pars["create_compile_model"],
            records_per_reference_model,
            reference_train_data,
            reference_train_labels_cat,
            filter_pars["reference_epochs"],
            filter_pars["reference_batch_size"],
        )
        # -- Save Step 2
        if save_path is not None:
            for i, model in enumerate(reference_models):
                path = save_path + "/reference_model" + str(i)
                logger.info(f"Save reference model: {path}.")
                model.save(path)
    else:
        # -- Or Load Step 2
        paths = load_pars["reference_models"]
        reference_models = []
        for path in paths:
            logger.info(f"Load reference model: {path}.")
            reference_models.append(models.load_model(path))

    # Steps 3-6
    if load or (
        "pairwise_distances_hlf_" + filter_pars["hlf_metric"]
        not in load_pars.keys()
    ):
        # -- Compute Step 3
        logger.info("Generate intermediate models")
        reference_intermediate_models = DirectGmia._gen_intermediate_models(
            reference_models, filter_pars["hlf_layer_number"]
        )

        # -- Compute Step 4
        logger.info("Extract reference high-level features.")
        reference_hlf = DirectGmia._extract_hlf(
            reference_intermediate_models, reference_train_data
        )
        # -- Save Step 4
        if save_path is not None:
            path = save_path + "/reference_hlf.npy"
            logger.info(f"Save reference high-level features: {path}.")
            np.save(path, reference_hlf)

        # -- Compute Step 5
        logger.info("Extract target high-level features.")
        target_hlf = DirectGmia._extract_hlf(
            reference_intermediate_models, attack_eval_data
        )
        # -- Save Step 5
        if save_path is not None:
            path = save_path + "/target_hlf.npy"
            logger.info(f"Save target high-level features: {path}.")
            np.save(path, reference_hlf)

        # -- Compute Step 6
        logger.info(
            "Compute pairwise distances between reference and target high-level "
            "features."
        )
        hlf_distances = DirectGmia._calc_pairwise_distances(
            target_hlf,
            reference_hlf,
            filter_pars["hlf_metric"],
        )
        # -- Save Step 6 (Results from step 3-6)
        if save_path is not None:
            path = (
                save_path
                + "/pairwise_distances_hlf_"
                + filter_pars["hlf_metric"]
                + ".npy"
            )
            logger.info(
                f"Save pairwise distances between reference and target high-level "
                f"features: {path}."
            )
            np.save(path, hlf_distances)

    else:
        # -- Or Load results from Steps 3-6
        path = load_pars["pairwise_distances_hlf_" + filter_pars["hlf_metric"]]
        logger.info(f"Load distance matrix : {path}.")
        hlf_distances = np.load(path)

    # -- Compute Step 7
    number_target_records = None
    if "number_target_records" in filter_pars.keys():
        number_target_records = filter_pars["number_target_records"]
    max_search_rounds = 100
    if "max_search_rounds" in filter_pars.keys():
        max_search_rounds = filter_pars["max_search_rounds"]
    neighbor_threshold = 0.5
    if "neighbor_threshold" in filter_pars.keys():
        neighbor_threshold = filter_pars["neighbor_threshold"]
    probability_threshold = 0.1
    if "probability_threshold" in filter_pars.keys():
        probability_threshold = filter_pars["probability_threshold"]

    logger.info("Determine potential vulnerable target records.")
    (
        target_records,
        neighbor_threshold,
        probability_threshold,
    ) = DirectGmia._select_target_records(
        len(reference_train_data),
        len(target_train_data),
        hlf_distances,
        neighbor_threshold,
        probability_threshold,
        number_target_records,
        max_search_rounds,
    )

    # Step 8
    filtered_target_indices = np.delete(data_conf["evaluation_indices"], target_records)

    return filtered_target_indices, neighbor_threshold, probability_threshold
