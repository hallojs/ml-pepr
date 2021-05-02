"""Direct generalized membership inference attack."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from pylatex import Command, NoEscape, Tabular, Figure, MiniPage, MultiColumn
from pylatex.section import Subsubsection
from pylatex.utils import bold
from sklearn.metrics import pairwise_distances

from statsmodels.distributions.empirical_distribution import ECDF

from scipy.interpolate import pchip
from tensorflow.keras.models import Model
from tensorflow.keras import models
import tensorflow as tf

from pepr import attack, report

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

plt.style.use("default")
# force line grid to be behind bar plots
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"


class DirectGmia(attack.Attack):
    """Direct Generalized Membership Inference Attack (d-GMIA)

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
    12. Evaluate the attack results.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the gmia.
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
        * number_target_records (int): If set, the selection algorithm performs
          `max_search_rounds`, to find a `neighbor_threshold`, that leads to a finding
          of `n_targets` target records. These target records are most vulnerable with
          respect to our selection criterion.
        * max_search_rounds (int): If `number_target_records` is given, maximal
          `max_search_rounds` are performed to find `number_target_records` of potential
          vulnerable target records.

    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
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
        * record_indices_per_target (numpy.ndarray): n*m array describing for all n
          target models which m indices where used in the training.

    target_models: iterable
        List of target models which should be tested.

    Attributes
    ----------
    attack_alias : str
        Alias for a specific instantiation of the attack class.
    attack_pars : dict
        Dictionary containing all needed parameters fo the attack.
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
    data_conf : dict
        Dictionary describing the data configuration of the given pentesting
        setting.
    target_models : iterable
        List of target models which should be tested.
    attack_results : dict

        * selected_target_records (numpy.ndarray): List of record indices selected as
          potential vulnerable.
        * neighbor_threshold (float): If distance is smaller then the neighbor threshold
          the record is selected as target record.
        * probability_threshold (float): For details see section 4.3 from the original
          publication.
        * reference_inferences (numpy.ndarray): Array of log losses of the predictions
          on the reference models.
        * target_inferences (numpy.ndarray): Array of log losses of the predictions on
          the target models.
        * used_target_records (numpy.ndarray): Target records finally used for the
          attack.
        * pchip_references (list): Interpolated ecdfs of sampled log losses.
        * ecdf_references (list): Estimated CDF of sampled log losses.
        * tp_list (list): True positives per cut-off-p-value (0, 0.01, 0.02, ..., 1) and
          target model.
        * fp_list (list): False positives per cut-off-p-value and target model.
        * fn_list (list): False negatives per cut-off-p-value and target model.
        * tn_list (list): True negatives per cut-off-p-value and target model.
        * precision_list (list): Attack precision per cut-off-p-value and target model.
        * recall_list (list): Attack recall per cut-off-p-value and target model.
        * overall_precision (list): Attack precision averaged over all target models per
          cut-off-p-value.
        * overall_recall (list): Attack recall averaged over all target models per
          cut-off-p-value.

    References
    ----------
    Implementation of the direct gmia from Long, Yunhui and Bindschaedler, Vincent and
    Wang, Lei and Bu, Diyue and Wang, Xiaofeng and Tang, Haixu and Gunter, Carl A and
    Chen, Kai (2018). Understanding membership inferences on well-generalized learning
    models. arXiv preprint arXiv:1802.04889.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        super().__init__(
            attack_alias, attack_pars, data, labels, data_conf, target_models
        )

        self.labels_cat = tf.keras.utils.to_categorical(
            labels, num_classes=attack_pars["number_classes"]
        )

        self.report_section = report.ReportSection(
            "Generalized Membership Inference Attack (Direct)",
            self.attack_alias,
            "gmia",
        )

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

        logger.debug(
            f"records_per_reference_model shape: "
            f"{records_per_reference_model.shape}"
        )

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
        number_target_records = None
        if "number_target_records" in self.attack_pars.keys():
            number_target_records = self.attack_pars["number_target_records"]
        max_search_rounds = 100
        if "max_search_rounds" in self.attack_pars.keys():
            max_search_rounds = self.attack_pars["max_search_rounds"]
        neighbor_threshold = 0.5
        if "neighbor_threshold" in self.attack_pars.keys():
            neighbor_threshold = self.attack_pars["neighbor_threshold"]
        probability_threshold = 0.1
        if "probability_threshold" in self.attack_pars.keys():
            probability_threshold = self.attack_pars["probability_threshold"]

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
        self.attack_results["selected_target_records"] = target_records
        self.attack_results["neighbor_threshold"] = neighbor_threshold
        self.attack_results["probability_threshold"] = probability_threshold

        # -- Compute Step 8
        logger.info("Infer log losses of reference models.")
        reference_inferences = DirectGmia._get_model_inference(
            target_records, attack_eval_data, attack_eval_labels_cat, reference_models
        )
        logger.debug(f"Reference inferences shape: {reference_inferences.shape}")
        self.attack_results["reference_inferences"] = reference_inferences

        # -- Compute Step 9
        logger.info("Infer log losses of target model.")
        target_inferences = DirectGmia._get_model_inference(
            target_records, attack_eval_data, attack_eval_labels_cat, self.target_models
        )
        logger.debug(f"Target inferences shape: {target_inferences.shape}")
        self.attack_results["target_inferences"] = target_inferences

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
        self.attack_results["used_target_records"] = used_target_records
        self.attack_results["pchip_references"] = pchip_references
        self.attack_results["ecdf_references"] = ecdf_references

        # -- Compute Step 11
        logger.info(
            "Determine members and non-members with left-tailed hypothesis test."
        )

        members, non_members = DirectGmia._hypothesis_test_runner(
            pchip_references,
            target_inferences,
            used_target_records,
        )

        # -- Compute Step 12
        records_per_target_model = self.data_conf["record_indices_per_target"]

        results = DirectGmia._attack_evaluation(
            members, non_members, records_per_target_model
        )

        # Union on dicts
        self.attack_results = {**self.attack_results, **results}

        logger.info(
            f"Attack precision over all target models per cut-off-p-value: "
            f'{self.attack_results["overall_precision"]}'
        )
        logger.info(
            f"Attack recall over all target models per cut-off-p-value: "
            f'{self.attack_results["overall_recall"]}'
        )

        # -- Output Attack Summary (for the first target model)
        res = self.attack_results
        # Number of selected target records which are members
        selected_target_records_in = len(
            set(res["selected_target_records"]).intersection(
                records_per_target_model[0]
            )
        )
        # Number of used target records which are members
        used_target_records_in = len(
            set(res["used_target_records"]).intersection(records_per_target_model[0])
        )

        logger.info(
            "Attack Summary"
            f"\n"
            f"\n###################### Attack Results ######################"
            f"\n"
            f"\n{'Cut-Off-P-Value:':<30}{'0.01':>10}{'0.02':>10}{'0.05':>10}"
            f"\n{'True Positives:':<30}"
            f"{res['tp_list'][1][0]:>10}"
            f"{res['tp_list'][2][0]:>10}"
            f"{res['tp_list'][5][0]:>10}"
            f"\n{'False Positives:':<30}"
            f"{res['fp_list'][1][0]:>10}"
            f"{res['fp_list'][2][0]:>10}"
            f"{res['fp_list'][5][0]:>10}"
            f"\n{'Precision:':<30}"
            f"{round(res['precision_list'][1][0], 3):>10}"
            f"{round(res['precision_list'][2][0], 3):>10}"
            f"{round(res['precision_list'][5][0], 3):>10}"
            f"\n{'Recall:':<30}"
            f"{round(res['recall_list'][1][0], 3):>10}"
            f"{round(res['recall_list'][2][0], 3):>10}"
            f"{round(res['recall_list'][5][0], 3):>10}"
            f"\n"
            f"\n{'Selected Target Records:':<30}"
            f"{len(res['selected_target_records']):>10}"
            f" ({selected_target_records_in} members)"
            f"\n{'Used Target Records:':<30}"
            f"{len(res['used_target_records']):>10}"
            f" ({used_target_records_in} members)"
        )

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
                f"Progress: Train reference model {i+1}/"
                f"{len(records_per_reference_model)}."
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
    def _gen_intermediate_models(source_models, layer_number):
        """Generate intermediate models.

        This intermediate models are used later to extract the high level features
        of the target and reference models.

        Parameters
        ----------
        source_models : numpy.ndarray
            Array of models from which the intermediate models should be extracted.
        layer_number : int
            Number of the intermediate layer used as the new output layer in the
            intermediate models.

        Returns
        -------
        numpy.ndarray
            Array of intermediate models.
        """
        if not isinstance(source_models, list):
            # noinspection PyProtectedMember
            layer = source_models.layers[layer_number]._name
            layer_output = source_models.get_layer(layer).output
            intermediate_model = Model(inputs=source_models.input, outputs=layer_output)
            return intermediate_model

        intermediate_models = np.array([])
        for i, model in enumerate(source_models):
            # noinspection PyProtectedMember
            layer = model.layers[layer_number]._name
            layer_output = model.get_layer(layer).output
            # noinspection PyTypeChecker
            intermediate_models = np.append(
                intermediate_models, Model(inputs=model.input, outputs=layer_output)
            )
        logger.info(f"Generated {len(source_models)} intermediate models.")
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

    @staticmethod
    def _select_target_records(
        background_knowledge_size,
        training_set_size,
        distances,
        neighbor_threshold=0.5,
        probability_threshold=0.1,
        number_target_records=None,
        max_search_rounds=100,
    ):
        """Select vulnerable target records.

        A record is selected as target record if it has few neighbours regarding its
        high level features. We estimate the number of neighbours of a record in the
        target training set over the number of neighbours in the reference
        training sets.

        Parameters
        ----------
        background_knowledge_size : int
            Size of the background knowledge of the attacker.
        training_set_size : int
            Number of samples used to train target model.
        distances : numpy.ndarray
            Distance array used for target selection.
        neighbor_threshold : float
            If distance is smaller then the neighbor threshold the record is selected
            as target record.
        probability_threshold : float
            For details see section 4.3 from the original publication.
        number_target_records : int
            If set, the selection algorithm performs `max_search_rounds`, to find a
            `neighbor_threshold`, that leads to a finding of `n_targets` target records.
            These target records are most vulnerable with respect to our selection
            criterion.
        max_search_rounds : int
            If `number_target_records` is given, maximal `max_search_rounds` are
            performed to find `number_target_records` of potential vulnerable target
            records.
        Returns
        -------
        numpy.ndarray
            Selected target records
        """

        def selection():
            # Number of neighbors in the reference data sets
            n_neighbors = np.count_nonzero(distances < neighbor_threshold, axis=1)
            # Estimated number of records in the training dataset of the target model
            est_n_neighbors = n_neighbors * (
                background_knowledge_size / training_set_size
            )
            records = np.where(est_n_neighbors < probability_threshold)[0]
            return records

        if number_target_records is None:
            return selection(), neighbor_threshold, probability_threshold

        target_records = selection()
        cnt = 0
        last_jmp = neighbor_threshold
        while len(target_records) != number_target_records and cnt < max_search_rounds:
            if cnt == 0 and len(target_records) < number_target_records:
                neighbor_threshold += last_jmp
            elif cnt == 0 and len(target_records) > number_target_records:
                neighbor_threshold += last_jmp
            else:
                if len(target_records) < number_target_records:
                    last_jmp *= 2
                    neighbor_threshold -= last_jmp
                else:
                    last_jmp /= 2
                    neighbor_threshold += last_jmp
            cnt += 1
            target_records = selection()
            if cnt % 5 == 0:
                logger.debug(f"Performed {cnt} search rounds.")
                logger.debug(f"Found {len(target_records)} target records.")
                logger.debug(f"Neighbor threshold: {neighbor_threshold}")

        logger.info(f"Performed {cnt} search rounds.")
        logger.info(f"Number of target records: {len(target_records)}.")
        logger.info(f"Target records (indexes): {target_records}.")
        logger.info(f"Neighbor threshold: {neighbor_threshold}")

        return target_records, neighbor_threshold, probability_threshold

    @staticmethod
    def _get_model_inference(idx_records, records, labels_cat, prediction_models):
        """Predict on trained models and calculate the log loss.

        Parameters
        ----------
        idx_records : numpy.ndarray
            Index array of records to predict on.
        records : numpy.ndarray
            Array of records used for prediction.
        labels_cat : numpy.ndarray
            Array of labels used to predict on (one-hot encoded).
        prediction_models : list
            Array of models used for prediction. It is also possible to pass only
            one model (not in an numpy.ndarray).

        Returns
        -------
        numpy.array
            Array of log losses of predictions.
        """
        if not isinstance(prediction_models, list):
            prediction_models = [prediction_models]
        inferences = []
        for i, model in enumerate(prediction_models):
            logger.info(f"Do inference on model {i+1}/{len(prediction_models)}.")
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
            Successfully used target records, smoothed ecdf, ecdf.
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
        pchip_references, target_inferences, used_target_records
    ):
        """Run hypothesis test on target inferences of multiple targets.

        Parameters
        ----------
        pchip_references : list
            Interpolated ecdfs of sampled log losses.
        target_inferences : numpy.ndarray
            Array of log losses of the predictions on the target models.
        used_target_records : numpy.ndarray
            Target records finally used for the attack.

        Returns
        -------
        tuple
            Two lists of lists of arrays. For each cut-off-p-value between 0 ...0.01
            ... 0.02, ...1 two lists of arrays, for each target model one array with the
            infered members and one array with the infered non-members.
        """
        members_list = []  # infered members per cut-of-p-value and per target model
        non_members_list = []

        for i in range(101):
            cut_off_p_value = i / 100

            members_list_p = []  # infered members for the i-th target model
            non_members_list_p = []

            for j, one_target_inferences in enumerate(target_inferences.T):
                members, non_members, _ = DirectGmia._hypothesis_test(
                    cut_off_p_value,
                    pchip_references,
                    np.array([one_target_inferences]).T,
                    used_target_records,
                )

                logger.debug(
                    f"Members determined for the {j}-th target model: " f"{members}"
                )

                members_list_p.append(members)
                non_members_list_p.append(non_members)

            members_list.append(members_list_p)
            non_members_list.append(non_members_list_p)

        return members_list, non_members_list

    @staticmethod
    def _attack_evaluation(
        infered_members, infered_non_members, records_per_target_model
    ):
        """Evaluate attack results.

        Parameters
        ----------
        infered_members : list
            List, that contains for each cut-off-p-value and target a numpy.ndarray of
            the infered members of the training data set.
        infered_non_members : list
            List, that contains for each cut-off-p-value and target a numpy.ndarray of
            the infered non-members of the training dataset.
        records_per_target_model : numpy.ndarray
            Array describing which target model is trained with which data records. The
            array has the shape: number-of-targets x number_of_attacked_target_records.

        Returns
        -------
        dict
            Dictionary containing the attack results.
        """
        tp_list = []
        fp_list = []
        fn_list = []
        tn_list = []
        precision_list = []
        recall_list = []
        overall_precision = []
        overall_recall = []

        # unpack lists per p-value
        for i in range(len(infered_members)):
            infered_members_p = infered_members[i]  # infered members per target model
            infered_non_members_p = infered_non_members[i]

            tp_list_p = []  # true positives per attacked target model
            fp_list_p = []
            fn_list_p = []
            tn_list_p = []
            precision_list_p = []
            recall_list_p = []

            # unpack lists per target model
            for j, members in enumerate(records_per_target_model):
                # count tps, etc.
                valid_infered_members = np.isin(infered_members_p[j], members)
                tp = np.count_nonzero(valid_infered_members)
                fp = len(valid_infered_members) - tp

                invalid_infered_non_members = np.isin(infered_non_members_p[j], members)
                fn = np.count_nonzero(invalid_infered_non_members)
                tn = len(invalid_infered_non_members) - fn

                tp_list_p.append(tp)
                fp_list_p.append(fp)
                fn_list_p.append(fn)
                tn_list_p.append(tn)

                precision = tp / (tp + fp) if (tp + fp) else 1
                recall = tp / (fn + tp) if (fn + tp) else 0
                precision_list_p.append(precision)
                recall_list_p.append(recall)

            # compute precision for all target models per cut-off-p-value
            number_of_target_models = len(infered_members_p)
            overall_precision_p = sum(precision_list_p) / number_of_target_models
            overall_recall_p = sum(recall_list_p) / number_of_target_models

            tp_list.append(tp_list_p)
            fp_list.append(fp_list_p)
            fn_list.append(fn_list_p)
            tn_list.append(tn_list_p)
            precision_list.append(precision_list_p)
            recall_list.append(recall_list_p)
            overall_precision.append(overall_precision_p)
            overall_recall.append(overall_recall_p)

        # attack results per cut-off-p-value and target model
        attack_results = {
            "tp_list": tp_list,
            "fp_list": fp_list,
            "fn_list": fn_list,
            "tn_list": tn_list,
            "precision_list": precision_list,
            "recall_list": recall_list,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
        }

        return attack_results

    def create_attack_report(self, save_path="gmia_report", pdf=False):
        """Create an attack report just for the given attack instantiation.

        Parameters
        ----------
        save_path : str
            Path to save the tex and pdf file of the attack report.
        pdf : bool
            If set, generate pdf out of latex file.
        """

        # Create directory structure for the attack report, including the figure
        # directory for the figures of the results subsubsection.
        os.makedirs(save_path + "/fig", exist_ok=True)

        self.create_attack_section(save_path=save_path)
        report.report_generator(save_path, [self.report_section], pdf)

    def create_attack_section(self, save_path):
        """Create a report section for the gmia attack instantiation."""

        self._report_attack_configuration()
        self._report_attack_results(save_path)

    def _report_attack_configuration(self):
        """Create subsubsection about the attack and data configuration."""
        # Create tables for attack parameters and the data configuration.
        ap = self.attack_pars
        dc = self.data_conf
        neighbor_threshold = str(round(self.attack_results["neighbor_threshold"], 5))
        probability_threshold = str(self.attack_results["probability_threshold"])
        if "number_target_records" in ap.keys():
            neighbor_threshold += " (auto)"
            probability_threshold += " (auto)"
        self.report_section.append(Subsubsection("Attack Details"))
        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                # -- Create table for the attack parameters.
                self.report_section.append(Command("centering"))
                with self.report_section.create(Tabular("|l|c|")) as tab_ap:
                    tab_ap.add_hline()
                    tab_ap.add_row(["Number of classes", ap["number_classes"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(
                        ["Number of reference models", ap["number_reference_models"]]
                    )
                    tab_ap.add_hline()
                    tab_ap.add_row(
                        [
                            "Reference training set size",
                            ap["reference_training_set_size"],
                        ]
                    )
                    tab_ap.add_hline()
                    tab_ap.add_row(["Reference epochs", ap["reference_epochs"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(["Reference batch size", ap["reference_batch_size"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(["HLF-Metric", ap["hlf_metric"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(["HLF-Layer-Number", ap["hlf_layer_number"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(["Neighbor-Threshold", neighbor_threshold])
                    tab_ap.add_hline()
                    tab_ap.add_row(["Probability-Threshold", probability_threshold])
                    tab_ap.add_hline()
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "table",
                        extra_arguments="Attack parameters (neighbor_threshold is "
                        "rounded to 5 decimal places)",
                    )
                )

            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                # -- Create table for the data configuration
                self.report_section.append(Command("centering"))
                nr_targets, target_training_set_size = dc[
                    "record_indices_per_target"
                ].shape
                with self.report_section.create(Tabular("|l|c|")) as tab_dc:
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        [
                            NoEscape("Attackers background-knowledge size ($A$)"),
                            len(dc["reference_indices"]),
                        ]
                    )
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        [
                            NoEscape("Samples used to train target models ($B$)"),
                            len(dc["target_indices"]),
                        ]
                    )
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        [
                            "Samples used to evaluate the attack",
                            len(dc["evaluation_indices"]),
                        ]
                    )
                    tab_dc.add_hline()
                    tab_dc.add_row(["Attacked target models", nr_targets])
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        ["Target model's training sets size", target_training_set_size]
                    )
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        [
                            NoEscape("Size of $A \cap B$"),
                            len(
                                set(dc["reference_indices"]) & set(dc["target_indices"])
                            ),
                        ]
                    )
                    tab_dc.add_hline()
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "table",
                        extra_arguments="Target and Data Configuration",
                    )
                )

    def _report_attack_results(self, save_path):
        """Create subsubsection describing the most important results of the attack.

        This subsection contains results only for the first target model.
        """
        self.report_section.append(Subsubsection("Attack Results"))

        # Precision-Recall Curve of the first target model
        precision = np.array(self.attack_results["precision_list"])
        recall = np.array(self.attack_results["recall_list"])

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(precision[:, 0], recall[:, 0])
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        fig.savefig(save_path + f"/fig/{alias_no_spaces}-precision_recall_curve.pdf")
        plt.close(fig)

        res = self.attack_results

        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                self.report_section.append(
                    Command(
                        "includegraphics",
                        NoEscape(f"fig/{alias_no_spaces}-precision_recall_curve.pdf"),
                        "width=8cm",
                    )
                )
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "figure",
                        extra_arguments="Precision-Recall Curve (for Selected Target Records)",
                    )
                )
            self.report_section.append(Command("hfill"))

            # Number of selected target records which are members
            selected_target_records_in = len(
                set(res["selected_target_records"]).intersection(
                    self.data_conf["record_indices_per_target"][0]
                )
            )
            # Number of used target records which are members
            used_target_records_in = len(
                set(res["used_target_records"]).intersection(
                    self.data_conf["record_indices_per_target"][0]
                )
            )

            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                with self.report_section.create(Tabular("|l|c|c|c|")) as result_tab:
                    result_tab.add_hline()
                    result_tab.add_row(
                        list(map(bold, ["Cut-Off-P-Value", "0.01", "0.02", "0.05"]))
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        [
                            "True Positives",
                            res["tp_list"][1][0],
                            res["tp_list"][2][0],
                            res["tp_list"][5][0],
                        ]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        [
                            "False Positives",
                            res["fp_list"][1][0],
                            res["fp_list"][2][0],
                            res["fp_list"][5][0],
                        ]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        [
                            "Precision",
                            round(res["precision_list"][1][0], 3),
                            round(res["precision_list"][2][0], 3),
                            round(res["precision_list"][5][0], 3),
                        ]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        [
                            "Recall",
                            round(res["recall_list"][1][0], 3),
                            round(res["recall_list"][2][0], 3),
                            round(res["recall_list"][5][0], 3),
                        ]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        "Selected Target Records",
                        MultiColumn(
                            3,
                            align="|c|",
                            data=f"{len(res['selected_target_records'])} "
                            f"({selected_target_records_in} members)",
                        ),
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        "Used Target Records",
                        MultiColumn(
                            3,
                            align="|c|",
                            data=f"{len(res['used_target_records'])} "
                            f"({used_target_records_in} members)",
                        ),
                    )
                    result_tab.add_hline()
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command("captionof", "table", extra_arguments="Attack Results")
                )

        selected_t = res["selected_target_records"]
        selected_t = self.labels[selected_t]
        used_t = res["used_target_records"]
        used_t = self.labels[used_t]
        classes = self.attack_pars["number_classes"]

        fig = plt.figure()
        ax = plt.axes()
        ax.set_xticks([i for i in range(classes)])
        ax.hist(selected_t, bins=np.arange(classes + 1) - 0.5, edgecolor="black")
        ax.hist(
            used_t,
            bins=np.arange(classes + 1) - 0.5,
            histtype="step",
            edgecolor="black",
        )
        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        fig.savefig(save_path + f"/fig/{alias_no_spaces}-hist_selected_records.pdf")
        plt.close(fig)

        with self.report_section.create(Figure(position="H")) as fig:
            fig.add_image(
                f"fig/{alias_no_spaces}-hist_selected_records.pdf", width=NoEscape(r"0.5\textwidth")
            )
            self.report_section.append(Command("captionsetup", "labelformat=empty"))
            self.report_section.append(
                Command(
                    "captionof",
                    "figure",
                    extra_arguments="Selected potential vulnerable target records per "
                    "class",
                )
            )
