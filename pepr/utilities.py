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


def filter_outlier(data, labels, filter_pars, save_path=None, load_pars=None):
    """
    Filter out potentially vulnerable samples.

    The general idea is similar to (:meth:`pepr.privacy.gmia.DirectGmia`) to find
    potential vulnerable records.

    Steps:

    1. Create mapping of records to reference models.
    2. Train the reference models.
    3. Generate intermediate models.
    4. Extract high-level features.
    5. Compute pairwise distances between high-level features.
    6. Determine outlier records.
    7. Remove outlier records from the dataset.

    Parameters
    ----------
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.\
    filter_pars : dict
        Dictionary containing needed filter parameters:

        * number_classes (int): Number of different classes the dataset.
        * number_reference_models (int): Number of reference models to be trained.
        * reference_training_set_size (int): Size of the trainings set for each
          reference model.
        * create_compile_model (function): Function that returns a compiled
          TensorFlow model (in gmia this is typically identical to the target model)
          used in the training of the reference models.
        * reference_epochs (int): Number of training epochs of the reference models.
        * reference_batch_size (int): Batch size used in the training of the
          reference models.
        * hlf_metric (str): Metric (typically 'cosine') used for the distance
          calculations in the high-level feature space. For valid metrics see
          documentation of sklearn.metrics.pairwise_distances.
        * hlf_layer_number (int): If value is n, the n-th layer of the model
          returned by create_compile_model is used to extract the high-level feature
          vectors.
        * distance_neighbor_threshold (float): If distance is smaller than the neighbor
          threshold the record is selected as target record.
        * number_neighbor_threshold (float): If number of neighbors of a record is
          smaller than this, it is considered a vulnerable example.
        * number_outlier (int): If set, the selection algorithm performs
          `max_search_rounds`, to find a `distance_neighbor_threshold`, that leads to a
          finding of `n_targets` target records. These target records are most
          vulnerable with respect to our selection criterion.
        * max_search_rounds (int): If `number_target_records` is given, maximal
          `max_search_rounds` are performed to find `number_target_records` of potential
          vulnerable target records.

    save_path : str
        If path is given, the following (partly computational expensive)
        intermediate results are saved to disk:

        * The mapping of training-records to reference models
        * The trained reference models
        * The high-level features
        * The matrix containing all pairwise distances between the high-level features.

    load_pars : dict
        If this dictionary is given, the following computational intermediate
        results can be loaded from disk.

        * records_per_reference_model (str) : Path to the mapping.
        * reference_models (list) : List of paths to the reference models.
        * pairwise_distance_hlf_<hlf_metric> (str) :  Path to the pairwise distance
          matrix between the high-level features using a hlf_metric (e.g. cosine).

    Returns
    -------
    numpy.ndarray:
        Dataset indices without outliers.
    float:
        Calculated neighbor distance threshold of the result.
    float:
        Calculated neighbor count threshold of the result.

    References
    ----------
    Partial implementation of the direct gmia from Long, Yunhui and Bindschaedler,
    Vincent and Wang, Lei and Bu, Diyue and Wang, Xiaofeng and Tang, Haixu and Gunter,
    Carl A and Chen, Kai (2018). Understanding membership inferences on well-generalized
    learning models. arXiv preprint arXiv:1802.04889.
    """
    from pepr.privacy.gmia import DirectGmia
    from tensorflow.keras import models, utils

    labels_cat = utils.to_categorical(labels, num_classes=filter_pars["number_classes"])

    load = load_pars is None

    # Step 1
    if load or "records_per_reference_model" not in load_pars.keys():
        # -- Compute Step 1
        logger.info("Create mapping of records to reference models.")
        records_per_reference_model = DirectGmia._assign_records_to_reference_models(
            filter_pars["number_reference_models"],
            len(data),
            filter_pars["reference_training_set_size"],
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
        f"records_per_reference_model shape: " f"{records_per_reference_model.shape}"
    )

    # Step 2
    if load or "reference_models" not in load_pars.keys():
        # -- Compute Step 2
        reference_models = DirectGmia._train_reference_models(
            filter_pars["create_compile_model"],
            records_per_reference_model,
            data,
            labels_cat,
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
        "pairwise_distances_hlf_" + filter_pars["hlf_metric"] not in load_pars.keys()
    ):
        # -- Compute Step 3
        logger.info("Generate intermediate models")
        reference_intermediate_models = DirectGmia._gen_intermediate_models(
            reference_models, filter_pars["hlf_layer_number"]
        )

        # -- Compute Step 4
        logger.info("Extract high-level features.")
        hlf = DirectGmia._extract_hlf(reference_intermediate_models, data)
        # -- Save Step 4
        if save_path is not None:
            path = save_path + "/hlf.npy"
            logger.info(f"Save high-level features: {path}.")
            np.save(path, hlf)

        # -- Compute Step 5
        logger.info("Compute pairwise distances between high-level features.")
        hlf_distances = DirectGmia._calc_pairwise_distances(
            hlf,
            hlf,
            filter_pars["hlf_metric"],
        )
        # -- Save Step 5 (Results from step 3-5)
        if save_path is not None:
            path = (
                save_path
                + "/pairwise_distances_hlf_"
                + filter_pars["hlf_metric"]
                + ".npy"
            )
            logger.info(f"Save pairwise distances between high-level features: {path}.")
            np.save(path, hlf_distances)

    else:
        # -- Or Load results from Steps 3-5
        path = load_pars["pairwise_distances_hlf_" + filter_pars["hlf_metric"]]
        logger.info(f"Load distance matrix : {path}.")
        hlf_distances = np.load(path)

    # -- Compute Step 6
    number_outlier = None
    if "number_outlier" in filter_pars.keys():
        number_outlier = filter_pars["number_outlier"]
    max_search_rounds = 100
    if "max_search_rounds" in filter_pars.keys():
        max_search_rounds = filter_pars["max_search_rounds"]
    distance_neighbor_threshold = 0.5
    if "distance_neighbor_threshold" in filter_pars.keys():
        distance_neighbor_threshold = filter_pars["distance_neighbor_threshold"]
    number_neighbor_threshold = 10
    if "number_neighbor_threshold" in filter_pars.keys():
        number_neighbor_threshold = filter_pars["number_neighbor_threshold"]

    logger.info("Determine potential vulnerable data records (outliers).")

    def select_outlier(
        distances,
        distance_neighbor_threshold=0.5,
        number_neighbor_threshold=10,
        number_outlier=None,
        max_search_rounds=100,
    ):
        def selection():
            n_neighbors = np.count_nonzero(
                distances < distance_neighbor_threshold, axis=1
            )
            return np.where(n_neighbors < number_neighbor_threshold)[0]

        if number_outlier is None:
            return selection(), distance_neighbor_threshold, number_neighbor_threshold

        outlier_indices = selection()
        # Calculating search area for binary search, so that every possible threshold
        # can be reached. The search area will look something like this:
        #                  initial jump size
        #                 |---------------|
        # [00000000000----*---------------] (search area)
        #             ^   ^              ^
        #             |   |              max distance value
        #             |   initial neighbor threshold
        #             min distance value
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        jmp_size = max(
            abs(distance_neighbor_threshold - max_dist),
            abs(distance_neighbor_threshold - min_dist),
        )
        cnt = 0
        while len(outlier_indices) != number_outlier and cnt < max_search_rounds:
            # Binary search
            jmp_size /= 2
            if len(outlier_indices) < number_outlier:
                distance_neighbor_threshold -= jmp_size
            else:
                distance_neighbor_threshold += jmp_size

            cnt += 1
            outlier_indices = selection()
            if cnt % 5 == 0:
                logger.debug(f"Performed {cnt} search rounds.")
                logger.debug(f"Found {len(outlier_indices)} outlier.")
                logger.debug(
                    f"Distance neighbor threshold: {distance_neighbor_threshold}"
                )

        logger.info(f"Performed {cnt} search rounds.")
        logger.info(f"Number of outliers: {len(outlier_indices)}.")
        logger.info(f"Outlier (indexes): {outlier_indices}.")
        logger.info(f"Distance neighbor threshold: {distance_neighbor_threshold}")

        return outlier_indices, distance_neighbor_threshold, number_neighbor_threshold

    (
        outlier_indices,
        distance_neighbor_threshold,
        number_neighbor_threshold,
    ) = select_outlier(
        hlf_distances,
        distance_neighbor_threshold,
        number_neighbor_threshold,
        number_outlier,
        max_search_rounds,
    )

    # Step 7
    filtered_data_indices = np.delete(np.arange(len(data)), outlier_indices)

    return filtered_data_indices, distance_neighbor_threshold, number_neighbor_threshold
