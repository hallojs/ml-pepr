"""Direct generalized membership inference attack.

Implementation of the direct gmia from Long, Yunhui and Bindschaedler, Vincent and Wang,
Lei and Bu, Diyue and Wang, Xiaofeng and Tang, Haixu and Gunter, Carl A and Chen, Kai
(2018). Understanding membership inferences on well-generalized learning models. arXiv
preprint arXiv:1802.04889.

Attack-Steps:

1. Create mapping of records to reference models.
2. Train the reference models.
3. Generate intermediate models.
4. Extract reference high-level features.
5. Extract target high-level features.
6. Compute pairwise distances between reference and target high-level
   features.
7. Determine potential vulnerable target records.
8. Infer log losses of reference models.
9. Infer log losses of target model.
10. Sample reference losses, approximate empirical cumulative distribution
    function, smooth ecdf with piecewise cubic interpolation.
11. Determine members and non-members with left-tailed hypothesis test.
"""
import logging
import numpy as np
from sklearn.metrics import pairwise_distances

from statsmodels.distributions.empirical_distribution import ECDF

from scipy.interpolate import pchip
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import models

from pepr.report.report_section import ReportSection

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DirectGmia:
    """Whole functionality needed for the direct gmia."""

    attack_pars: dict
    data: np.ndarray
    labels: np.ndarray
    data_conf: dict
    target_models: list
    report_section: ReportSection
    save_path: str  # TODO: Change this to pythons path object
    load_pars: dict
    end_results: dict

    # TODO: Precise neighbor_threshold, probability_threshold

    def __init__(self, attack_pars, data, labels, data_conf, target_models):
        """Initialize direct generalized membership inference attack.

        Parameters
        ----------
        attack_pars : dict
            Dictionary containing all needed attack parameters:

            * number_classes (int): Number of different classes the target model
              predict.
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
            * cut_off_p_value (float): Cut-off-p-value used for the hypothesis test.

        data : numpy.ndarray
            Dataset with all training samples used in this pentesting setting.
        labels : numpy.ndarray
            Array of all labels used in this pentesting setting.
        data_conf: dict
            Dictionary describing which record-indices are used to train the reference
            models, the target model(s) and which are used for the evaluation of the
            attack.

            * reference_indices (list): List of indices describing which of the records
              from data are used to train the reference models.
            * target_indices (list): List of indices describing which of the records
              from data were used to train the target model(s).
            * evaluation_indices (list): List of indices describing which of the records
              from data are used to evaluate the attack. Typically these are to one half
              records used to train the target models and one half neither used to train
              the target model(s) or the reference models.

        target_models: list
            List of target models which should be tested.
        """
        self.attack_pars = attack_pars
        self.data = data
        self.labels = labels
        self.labels_cat = tf.keras.utils.to_categorical(
            labels, num_classes=attack_pars["number_classes"]
        )
        self.data_conf = data_conf
        self.target_models = target_models

    def run(self, save_path=None, load_pars=None):
        """Run the direct generalized membership inference attack.

        Parameters
        ----------
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
        """

        # Slice data set
        # -- Used to train the reference models
        reference_train_data = self.data[self.data_conf["reference_indices"]]
        # reference_train_labels = self.labels[self.data_conf["reference_indices"]]
        reference_train_labels_cat = self.labels_cat[
            self.data_conf["reference_indices"]
        ]

        # -- Used to train the target models
        target_train_data = self.data[self.data_conf["target_indices"]]
        # target_train_labels = self.labels[self.data_conf["target_indices"]]
        # target_train_labels_cat = self.labels_cat[self.data_conf["target_indices"]]

        # -- Used for the evaluation of the attack
        attack_eval_data = self.data[self.data_conf["evaluation_indices"]]
        # attack_eval_labels = self.labels[self.data_conf["evaluation_indices"]]
        attack_eval_labels_cat = self.labels_cat[self.data_conf["evaluation_indices"]]

        load = load_pars is None

        # Step 1
        if load or "records_per_reference_model" not in load_pars.keys():
            # -- Compute Step 1
            logger.info("Create mapping of records to reference models.")
            records_per_reference_model = (
                DirectGmia._assign_records_to_reference_models(
                    self.attack_pars["number_reference_models"],
                    len(reference_train_data),
                    self.attack_pars["reference_training_set_size"],
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

        # Step 2
        if load or "reference_models" not in load_pars.keys():
            # -- Compute Step 2
            reference_models = DirectGmia._train_reference_models(
                self.attack_pars["create_compile_model"],
                records_per_reference_model,
                reference_train_data,
                reference_train_labels_cat,
                self.attack_pars["reference_epochs"],
                self.attack_pars["reference_batch_size"],
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
            "pairwise_distances_hlf_" + self.attack_pars["hlf_metric"]
            not in load_pars.keys()
        ):
            # -- Compute Step 3
            logger.info("Generate intermediate models")
            reference_intermediate_models = DirectGmia._gen_intermediate_models(
                reference_models, self.attack_pars["hlf_layer_number"]
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
                self.attack_pars["hlf_metric"],
            )
            # -- Save Step 6 (Results from step 3-6)
            if save_path is not None:
                path = (
                    save_path
                    + "/pairwise_distances_hlf_"
                    + self.attack_pars["hlf_metric"]
                    + ".npy"
                )
                logger.info(
                    f"Save pairwise distances between reference and target high-level "
                    f"features: {path}."
                )
                np.save(path, hlf_distances)

        else:
            # -- Or Load results from Steps 3-6
            path = load_pars["pairwise_distances_hlf_" + self.attack_pars["hlf_metric"]]
            logger.info(f"Load distance matrix : {path}.")
            hlf_distances = np.load(path)

        # -- Compute Step 7
        logger.info("Determine potential vulnerable target records.")
        target_records = DirectGmia._select_target_records(
            self.attack_pars["neighbor_threshold"],
            self.attack_pars["probability_threshold"],
            len(reference_train_data),
            len(target_train_data),
            hlf_distances,
        )

        # -- Compute Step 8
        logger.info("Infer log losses of reference models.")
        reference_inferences = DirectGmia._get_model_inference(
            target_records, attack_eval_data, attack_eval_labels_cat, reference_models
        )
        logger.debug(f"Reference inferences shape: {reference_inferences.shape}")

        # -- Compute Step 9
        logger.info("Infer log losses of target model.")
        target_inferences = DirectGmia._get_model_inference(
            target_records, attack_eval_data, attack_eval_labels_cat, self.target_models
        )
        logger.debug(f"Target inferences shape: {target_inferences.shape}")

        # -- Compute Step 10
        logger.info(
            "Sample reference losses, approximate empirical cumulative distribution "
            "function, smooth ecdf with piecewise cubic interpolation."
        )
        (
            used_target_records,
            pchip_references,
            ecdf_references,
        ) = DirectGmia._sample_reference_losses(target_records, reference_inferences)
        logger.debug(f"PCHIP references shape: {len(pchip_references)}")
        logger.debug(f"ECDF references shape: {len(ecdf_references)}")

        # -- Compute Step 11
        logger.info(
            "Determine members and non-members with left-tailed hypothesis test."
        )
        # TODO: Extend evaluation to multiple targets
        DirectGmia._hypothesis_test_runner(
            self.attack_pars["cut_off_p_value"],
            pchip_references,
            target_inferences,
            used_target_records,
        )
        # members, non_members, p_values = DirectGmia._hypothesis_test(
        #     self.attack_pars["cut_off_p_value"],
        #     pchip_references,
        #     target_inferences,
        #     used_target_records,
        # )
        #
        # # -- Save end results
        # self.end_results = {
        #     "members": members,
        #     "non_members": non_members,
        #     "p_values": p_values,
        # }
        # logger.info(f"Determined training set members of the target model: {members}.")
        # logger.info(
        #     f"Determined training set non-members of the target model: "
        #     f"{non_members}."
        # )
        # logger.info(f"P-values of the final hypothesis-test: {p_values}.")

    @staticmethod
    def _assign_records_to_reference_models(
        number_reference_models, background_knowledge_size, training_set_size
    ):
        """Create training datasets for the reference models.

        Parameters
        ----------
        number_reference_models : int
            Number of reference models to be trained.
        background_knowledge_size : int
            Size of the background knowledge of the attacker.
        training_set_size : int
            Number of samples used to train each reference model.

        Returns
        -------
        numpy.ndarray
            Array that describes which reference model should be trained with which
            samples.
        """
        records_per_reference_model = np.array([])
        for i in range(number_reference_models):
            np.random.seed(i)
            # sampling
            idx = np.random.choice(
                background_knowledge_size, training_set_size, replace=False
            )
            if i > 0:
                records_per_reference_model = np.append(
                    records_per_reference_model, [idx], axis=0
                )
            else:
                records_per_reference_model = [idx]

        return records_per_reference_model

    @staticmethod
    def _train_reference_models(
        create_compile_model,
        records_per_reference_model,
        train_data,
        train_labels,
        epochs,
        batch_size,
    ):
        """Train reference models based on a compiled TensorFlow Keras model.

        Parameters
        ----------
        create_compile_model : function
            Return compiled TensorFlow Keras model, used to train the reference
            models.
        records_per_reference_model : np.ndarray
            Describes which record is to use to train which model.
        train_data : numpy.ndarray
            Training data for the reference models.
        train_labels : numpy.ndarray
            Training labels (one-hot encoding) for the reference models.
        epochs : int
            Number of training epochs for each reference model.
        batch_size : int
            Size of mini batches used during the training.

        Returns
        -------
        numpy.ndarray
            Array of the trained reference models.
        """
        reference_models = []
        for i, records in enumerate(records_per_reference_model):
            logger.info(
                f"Progress: Train reference model {i}/"
                f"{records_per_reference_model}."
            )
            tmp_model = create_compile_model()
            tmp_model.fit(
                train_data[records],
                train_labels[records],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            reference_models.append(tmp_model)

        return reference_models

    @staticmethod
    def _load_models(path, number_models):
        """Load trained and saved models.

        Parameters
        ----------
        path : str
            Path to saved models.
        number_models : int
            Describes how many models should be loaded.

        Returns
        -------
        numpy.ndarray
            Array of loaded models.
        """
        all_models = np.array([])
        for i in range(number_models):
            logger.info(f"Progress: Load reference model {i+1}/{number_models}.")
            model = models.load_model(path + str(i))
            all_models = np.append(all_models, model)

        return all_models

    # TODO: Do not use intermediate models and grab the output of the
    #  intermediate layers directly from the reference models.

    @staticmethod
    def _gen_intermediate_models(models, layer_number):
        """Generate intermediate models.

        This intermediate models are used later to extract the high level features
        of the target and reference models.

        Parameters
        ----------
        models : numpy.ndarray
            Array of models from which the intermediate models should be extracted.
        layer_number : int
            Number of the intermediate layer used as the new output layer in the
            intermediate models.

        Returns
        -------
        numpy.ndarray
            Array of intermediate models.
        """
        if not isinstance(models, list):
            # noinspection PyProtectedMember
            layer = models.layers[layer_number]._name
            layer_output = models.get_layer(layer).output
            intermediate_model = Model(inputs=models.input, outputs=layer_output)
            return intermediate_model

        intermediate_models = np.array([])
        for i, model in enumerate(models):
            # noinspection PyProtectedMember
            layer = model.layers[layer_number]._name
            layer_output = model.get_layer(layer).output
            # noinspection PyTypeChecker
            intermediate_models = np.append(
                intermediate_models, Model(inputs=model.input, outputs=layer_output)
            )
        logger.info(f"Generated {len(models)} intermediate models.")
        return intermediate_models

    # TODO: Add more precise description about the extraction process to the docstring.

    @staticmethod
    def _extract_hlf(intermediate_models, data):
        """Extract the high level features from the intermediate models.

        For details see paper section 4.3.

        Parameters
        ----------
        intermediate_models : numpy.ndarray
            Array of intermediate models. It is also possible to pass a single
            model.
        data : numpy.ndarray
            Data from which the high-level features should be extracted.

        Returns
        -------
        numpy.ndarray
            Array of high level features. One feature vector for each data sample.
        """
        if type(intermediate_models) != np.ndarray:
            intermediate_models = np.array([intermediate_models])

        feature_vecs = np.empty((len(data), 0))
        for i, model in enumerate(intermediate_models):
            predictions = model.predict(data)
            feature_vecs = np.append(feature_vecs, predictions, axis=1)
        logger.info(
            f"Extracted high-level-features from {len(intermediate_models)} "
            f"intermediate models."
        )
        return feature_vecs

    @staticmethod
    def _calc_pairwise_distances(features_target, features_reference, metric, n_jobs=1):
        """Calculate pairwise distances between given features.

        Parameters
        ----------
        features_target : numpy.ndarray
            First array for pairwise distances.
        features_reference : numpy.ndarray
            Second array for pairwise distances.
        metric : str
            Metric used for the distance calculations. For valid metrics see
            documentation of sklearn.metrics.pairwise_distances.
        n_jobs : int, optional
            Number of parallel computation jobs.

        Returns
        -------
        numpy.ndarray
            The distance between features_target[i] and features_reference[j] is
            saved in distances[i][j].

        """
        distances = pairwise_distances(
            features_target, features_reference, metric=metric, n_jobs=n_jobs
        )

        return distances

    # TODO: Precise parameter description.

    @staticmethod
    def _select_target_records(
        neighbor_threshold,
        probability_threshold,
        background_knowledge_size,
        training_set_size,
        distances,
    ):
        """Select vulnerable target records.

        A record is selected as target record if it has few neighbours regarding
        its high level features. We estimate the number of neighbours of a record
        in the target training set over the number of neighbours in the reference
        training sets.

        Parameters
        ----------
        neighbor_threshold : float
            If distance is smaller then the neighbor threshold the record is
            selected as target record.
        probability_threshold : float
            For details see section 4.3 from the paper.
        background_knowledge_size : int
            Size of the background knowledge of the attacker.
        training_set_size : int
            Number of samples used to train target model.
        distances : numpy.ndarray
            Distance array used for target selection.

        Returns
        -------
        numpy.ndarray
            Selected target records
        """
        logger.info("Start target record selection.")
        if np.min(distances) >= neighbor_threshold:
            logger.warning("neighbor_threshold is smaller then all distances!")

        n_neighbors = np.count_nonzero(distances < neighbor_threshold, axis=1)

        est_n_neighbors = n_neighbors * (background_knowledge_size / training_set_size)

        target_records = np.where(est_n_neighbors < probability_threshold)[0]

        logger.info(f"Minimal distance: {np.min(distances)}.")
        logger.info(f"Mean number of neighbors: {np.mean(n_neighbors)}.")
        logger.info(f"Mean estimates number of neighbors: {np.mean(est_n_neighbors)}.")
        logger.info(f"Number of target records: {len(target_records)}.")
        logger.info(f"Target records (indexes): {target_records}.")
        logger.info(
            f"Number of neighbors per target record: {n_neighbors[target_records]}."
        )

        return target_records

    @staticmethod
    def _get_model_inference(idx_records, records, labels_cat, models):
        """Predict on trained models and calculate the log loss.

        Parameters
        ----------
        idx_records : numpy.ndarray
            Index array of records to predict on.
        records : numpy.ndarray
            Array of records used for prediction.
        labels_cat : numpy.ndarray
            Array of labels used to predict on (one-hot encoded).
        models : list
            Array of models used for prediction. It is also possible to pass only
            one model (not in an numpy.ndarray).

        Returns
        -------
        numpy.array
            Array of log losses of predictions.
        """
        if not isinstance(models, list):
            models = [models]
        inferences = []
        for i, model in enumerate(models):
            logger.info(f"Do inference on model {i+1}/{len(models)}.")
            predictions = model.predict(records[idx_records])
            filter_predictions = np.max(predictions * labels_cat[idx_records], axis=1)
            inferences.append(-np.log(filter_predictions))

        return np.asarray(inferences).T

    @staticmethod
    def _sample_reference_losses(target_records, reference_inferences):
        """Sample reference log losses.

        Sample the log losses of a record regarding its label. Estimate the CDF of
        this samples and smooth the estimated CDF with the shape-preserving
        piecewise cubic interpolation.

        Parameters
        ----------
        target_records : numpy.ndarray
            Array of target records for sampling the reference log losses.
        reference_inferences : numpy.ndarray
            Array of log losses of the predictions on the reference models.

        Returns
        -------
        tuple
            Successfully used target records and smoothed ecdf.
        """
        # empirical cdf
        ecdf_references = []
        # piecewise cubic interpolation
        pchip_references = []

        used_target_records = []

        for idx in range(len(target_records)):
            ecdf_val = ECDF(reference_inferences[idx, :])
            ecdf_references.append(ecdf_val)

            try:
                pchip_val = pchip(ecdf_val.x[1:], ecdf_val.y[1:])
            except ValueError:
                continue

            used_target_records.append(target_records[idx])
            pchip_references.append(pchip_val)

        used_target_records = np.asarray(used_target_records)

        logger.info(f"Number of used target records: {len(used_target_records)}.")
        logger.info(f"Used target records (indexes): {used_target_records}.")

        return used_target_records, pchip_references, ecdf_references

    @staticmethod
    def _hypothesis_test(
        cut_off_p_value, pchip_references, target_inferences, used_target_records
    ):
        """Left-tailed hypothesis test for target inferences of a single target model.

        Parameters
        ----------
        cut_off_p_value : float
            Cut-off-p-value used for the hypothesis test.
        pchip_references : list
            Interpolated ecdfs of sampled log losses.
        target_inferences : numpy.ndarray
            Array of log losses of the predictions on the target models.
        used_target_records : numpy.ndarray
            Target records finally used for the attack.

        Returns
        -------
        tuple
            Array of members, array of non-members, array of calculated p-values.
        """
        # Calculate p-values
        p_values = []
        for idx in range(len(used_target_records)):
            p_values.append(pchip_references[idx](target_inferences[idx, :]))

        members = []
        non_members = []

        # Calculate membership
        for idx, target_record in enumerate(used_target_records):
            if p_values[idx] >= cut_off_p_value:
                non_members.append(target_record)
            else:
                members.append(target_record)

        return np.asarray(members), np.asarray(non_members), np.asarray(p_values)

    @staticmethod
    def _hypothesis_test_runner(
        cut_off_p_value, pchip_references, target_inferences, used_target_records
    ):
        """Run hypothesis test on target inferences of multiple targets.

        Parameters
        ----------
        cut_off_p_value : float
            Cut-off-p-value used for the hypothesis test.
        pchip_references : list
            Interpolated ecdfs of sampled log losses.
        target_inferences : numpy.ndarray
            Array of log losses of the predictions on the target models.
        used_target_records : numpy.ndarray
            Target records finally used for the attack.

        Returns
        -------
        tuple
            Array of members, array of non-members, array of calculated p-values.
        """
        for one_target_inferences in target_inferences.T:
            logger.debug(f'One target inferences shape: {one_target_inferences.shape}')
            members, non_members, p_values = DirectGmia._hypothesis_test(
                cut_off_p_value,
                pchip_references,
                np.array([one_target_inferences]).T,
                used_target_records,
            )
            logger.info(f"Members: {members}")
