"""Membership inference attack against machine learning models."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from pylatex import Command, NoEscape, Tabular, Figure, MiniPage
from pylatex.section import Subsubsection
from pylatex.utils import bold

from tensorflow.keras import models

from pepr import attack, report

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

plt.style.use("default")
# force line grid to be behind bar plots
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"


class Mia(attack.Attack):
    """
    Membership Inference Attacks (MIA) Against Machine Learning Models.

    Attack-Steps:

    1. Create dataset mapping for shadow models.
    2. Train shadow models.
    3. Generate attack model dataset.
    4. Train attack models.
    5. Evaluate attack models.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the mia class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * number_classes (int): Number of different classes the target model predicts.
        * number_shadow_models (int): Number of shadow models to be trained.
        * shadow_training_set_size (int): Size of the trainings set for each
          shadow model. The corresponding evaluation sets will have the same size.
        * create_compile_shadow_model (function): Function that returns a compiled
          TensorFlow model (typically identical to the target model) used in the
          training of the shadow models.
        * create_compile_attack_model (function): Function that returns a compiled
          TensorFlow model used for the attack models. The model output is expected to
          be a single floating-point value per prediction.
        * shadow_epochs (int): Number of training epochs of the shadow models.
        * shadow_batch_size (int): Batch size used in the training of the
          shadow models.
        * attack_epochs (int): Number of training epochs of the attack models.
        * attack_batch_size (int): Batch size used in the training of the
          attack models.

    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
    data_conf: dict
        Dictionary describing which record-indices are used to train the shadow
        models, the target model(s) and which are used for the evaluation of the
        attack.

        * shadow_indices (list): List of indices describing which of the records
          from data are used to train the shadow models.
        * target_indices (list): List of indices describing which of the records
          from data were used to train the target model(s).
        * evaluation_indices (list): List of indices describing which of the records
          from data are used to evaluate the attack.
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
        setting by specifying which record-indices are used to train the shadow
        models, the target model(s) and which are used for the evaluation of the
        attack.

        * shadow_indices (list): List of indices describing which of the records
          from data are used to train the shadow models.
        * target_indices (list): List of indices describing which of the records
          from data were used to train the target model(s).
        * evaluation_indices (list): List of indices describing which of the records
          from data are used to evaluate the attack.
        * record_indices_per_target (numpy.ndarray): n*m array describing for all n
          target models which m indices where used in the training.

    target_models : iterable
        List of target models which should be tested.
    attack_results : dict
        Dictionary storing the attack model results. A list "per attack model and
        target model" has the shape (attack model, target model) -> First index
        specifies the attack model, the second index the target model.

        * tp_list (numpy.ndarray): True positives per attack model and target model.
        * fp_list (numpy.ndarray): False positives per attack model and target model.
        * fn_list (numpy.ndarray): False negatives per attack model and target model.
        * tn_list (numpy.ndarray): True negatives per attack model and target model.
        * eval_accuracy_list (numpy.ndarray): Evaluation accuracy on evaluation records
          per attack model and target model.
        * precision_list (numpy.ndarray): Attack precision per attack model and target
          model.
        * recall_list (numpy.ndarray): Attack recall per attack model and target model.
        * eval_accuracy (numpy.ndarray): Evaluation accuracy averaged over all attack
          models per target model.
        * precision (numpy.ndarray): Attack precision averaged over all attack models
          per target model.
        * recall (numpy.ndarray): Attack recall averaged over all attack models per
          target model.
        * overall_eval_accuracy (float): Evaluation accuracy averaged over all target
          models.
        * overall_precision (float): Attack precision averaged over all target models.
        * overall_recall (float): Attack recall averaged over all target models.
        * shadow_train_accuracy_list (list): Accuracy on training records per shadow
          model and target model.
        * shadow_test_accuracy_list (list): Accuracy on test records per shadow model
          and target model.
        * shadow_train_accuracy (float): Accuracy on train records averaged over all
          shadow models per target model.
        * shadow_test_accuracy (float): Accuracy on test records averaged over all
          shadow models per target model.

    References
    ----------
    Implementation of the basic membership inference attack by Reza Shokri, Marco
    Stronati, Congzheng Song and Vitaly Shmatikov. Membership inference attacks
    against machine learning models 2017 IEEE Symposium on Security and Privacy (SP).
    IEEE, 2017.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        super().__init__(
            attack_alias, attack_pars, data, labels, data_conf, target_models
        )
        self.report_section = report.ReportSection(
            "Membership Inference Attack",
            self.attack_alias,
            "mia",
        )

    def run(self, save_path=None, load_pars=None):
        """
        Run membership inference attack.

        Parameters
        ----------
        save_path : str
            If path is given, the following (partly computational expensive)
            intermediate results are saved to disk:

            * The mapping of training-records to shadow models.
            * The trained shadow models.
            * The attack datasets for training the attack model.
            * The trained attack models.
        load_pars : dict
            If this dictionary is given, the following computational intermediate
            results can be loaded from disk.

            * shadow_data_indices (str) : Path to shadow data mapping.
            * shadow_models (list) : List of paths to shadow models.
            * attack_datasets (str) : Path to attack datasets.
            * attack_models (list) : List of paths to attack models.
        """

        load = load_pars != None
        save = save_path != None

        # Slice dataset
        shadow_train_data = self.data[self.data_conf["shadow_indices"]]
        shadow_train_labels = self.labels[self.data_conf["shadow_indices"]]

        attack_test_data = self.data[self.data_conf["evaluation_indices"]]
        attack_test_labels = self.labels[self.data_conf["evaluation_indices"]]

        # Step 1: Create dataset mapping for shadow models
        # shadow_data_indices[i]:
        #   Training:
        #       shadow_train_indices - index 0
        #   Test:
        #       shadow_test_indices - index 1
        if load and "shadow_data_indices" in load_pars.keys():
            path = load_pars["shadow_data_indices"]
            logger.info(f"Load mapping of records to shadow models: {path}.")
            shadow_data_indices = np.load(path)
        else:
            logger.info("Create mapping of records to shadow models.")
            shadow_data_indices = Mia._create_shadow_model_datasets(
                len(self.data_conf["shadow_indices"]),
                self.attack_pars["number_shadow_models"],
                self.attack_pars["shadow_training_set_size"],
            )
        if save:
            path = save_path + "/shadow_data_indices.npy"
            logger.info(f"Save mapping of records to shadow models: {path}.")
            np.save(path, shadow_data_indices)
        logger.debug(f"shadow_datasets shape: {shadow_data_indices.shape}")

        # Step 2: Train shadow models
        if load and "shadow_models" in load_pars.keys():
            paths = load_pars["shadow_models"]
            shadow_models = []
            for path in paths:
                logger.info(f"Load pre-trained shadow models: {path}.")
                shadow_models.append(models.load_model(path))
        else:
            logger.info("Train shadow models.")
            shadow_models = Mia._train_shadow_models(
                self.attack_pars["create_compile_shadow_model"],
                shadow_data_indices,
                shadow_train_data,
                shadow_train_labels,
                self.attack_pars["shadow_epochs"],
                self.attack_pars["shadow_batch_size"],
            )
        if save:
            for i, model in enumerate(shadow_models):
                path = save_path + "/shadow_model" + str(i)
                logger.info(f"Save trained shadow model: {path}.")
                model.save(path)

        # Step 3: Generate attack model dataset
        # attack_datasets[i]:
        #   "indices"
        #   "prediction_vectors"
        #   "attack_labels"
        if load and "attack_datasets" in load_pars.keys():
            path = load_pars["attack_datasets"]
            logger.info(f"Load attack model datasets: {path}.")
            attack_datasets = list(np.load(path, allow_pickle=True))
        else:
            logger.info("Generate attack dataset.")
            attack_datasets = Mia._generate_attack_dataset(
                shadow_models,
                shadow_data_indices,
                self.attack_pars["number_classes"],
                shadow_train_data,
                shadow_train_labels,
            )
        if save:
            path = save_path + "/attack_datasets.npy"
            logger.info(f"Save attack model datasets: {path}.")
            np.save(path, attack_datasets)

        # Step 4: Train attack models
        if load and "attack_models" in load_pars.keys():
            paths = load_pars["attack_models"]
            attack_models = []
            for path in paths:
                logger.info(f"Load pre-trained attack models: {path}.")
                attack_models.append(models.load_model(path))
        else:
            logger.info("Train attack models.")
            attack_models = Mia._train_attack_models(
                self.attack_pars["create_compile_attack_model"],
                attack_datasets,
                self.attack_pars["attack_epochs"],
                self.attack_pars["attack_batch_size"],
            )
        if save:
            for i, model in enumerate(attack_models):
                path = save_path + "/attack_model" + str(i)
                logger.info(f"Save trained attack model: {path}.")
                model.save(path)

        # Step 5: Evaluate attack models
        logger.info("Evaluate attack models.")

        # -- Generate target model dataset and true labels
        target_data = Mia._get_target_model_indices(
            self.data_conf["target_indices"],
            self.data_conf["evaluation_indices"],
            self.data_conf["record_indices_per_target"],
            attack_test_labels,
            self.attack_pars["number_classes"],
        )

        # -- Get target predictions for attack model input
        target_prediction = Mia._get_target_predictions(
            self.target_models, target_data, attack_test_data
        )

        # -- Evaluate
        logger.info("Calculate attack model summary.")
        self.attack_results = Mia._attack_model_evaluation(
            attack_models,
            target_prediction,
            target_data,
        )
        logger.info("Calculate shadow model summary.")
        self.attack_results.update(
            Mia._shadow_model_evaluation(
                shadow_models,
                shadow_data_indices,
                shadow_train_data,
                shadow_train_labels,
            )
        )

        # Print attack summary
        def _list_to_formatted_string(arr):
            string = ""
            for item in arr:
                string = string + f"{round(item, 3):>10}"
            return string

        logger.info(
            "Attack Summary"
            f"\n"
            f"\n###################### Shadow Results ######################"
            f"\n"
            + f"\n{'Shadow Models:':<30}"
            + f"\n{'Average Training Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["shadow_train_accuracy"])
            + f"\n{'Average Evaluation Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["shadow_test_accuracy"])
            + f"\n"
            f"\n###################### Attack Results ######################"
            f"\n"
            f"\n{'Attack Models:':<30}"
            + f"\n{'Average Test Accuracy:':<30}"
            + _list_to_formatted_string([self.attack_results["overall_eval_accuracy"]])
            + f"\n{'Average Precision:':<30}"
            + _list_to_formatted_string([self.attack_results["overall_precision"]])
            + f"\n{'Average Recall:':<30}"
            + _list_to_formatted_string([self.attack_results["overall_recall"]])
        )

    @staticmethod
    def _get_target_model_indices(
        target_indices,
        evaluation_indices,
        record_indices_per_target,
        attack_test_labels,
        number_classes,
    ):
        """
        Calculate classified indices per target with their true attack label for
        evaluation and convert indices to evaluation dataset space.

        Parameters
        ----------
        target_indices : list
            List of indices used to train all target models. Indices in complete
            dataset range.
        evaluation_indices : list
            List of indices for attack model evaluation. Indices in complete dataset
            range.
        record_indices_per_target : numpy.ndarray
            List of mapping which records of the target_indices are used to train a
            target model.
        attack_test_labels : numpy.ndarray
            List of labels in the evaluation dataset.
        number_classes : int
            Number of classes in the dataset.

        Returns
        -------
        dict
            Dictionary storing the classified indices and labels. Indices are in the
            evaluation data slice. The indices range starts with 0 and ends with the
            length of the evaluation data slice.

            * indices_per_target (list): Classified indices of the evaluation dataset
              per target model and per class. Shape: (target_models, classes)
            * true_attack_labels_per_target (list): The true labels for the attack
              evaluation per target model and per class. Shape: (target_models, classes)
        """
        eval_indices_target_classified = []  # shape = (target models, classes)
        true_labels_target_classified = []  # shape = (target models, classes)

        eval_i = np.arange(len(evaluation_indices))
        for i, record_indices in enumerate(record_indices_per_target):

            # Extract target model training data indices (full range)
            tm_train_origin = np.array(target_indices)[record_indices]
            # Convert indices from full range to evaluation data range
            e_train_i = np.nonzero(tm_train_origin[:, None] == evaluation_indices)[1]

            # Extract non-train indices aka. test data
            e_test_i = np.delete(eval_i, e_train_i)

            # Train and test indices should have the same length
            if len(e_test_i) > len(e_train_i):
                e_test_i = e_test_i[: len(e_train_i)]
            elif len(e_test_i) < len(e_train_i):
                e_train_i = e_train_i[: len(e_test_i)]

            evaluation_i = np.concatenate((e_train_i, e_test_i))
            attack_true_labels = np.in1d(evaluation_i, e_train_i)

            # Classify
            eval_indices_target_classified.append([])
            true_labels_target_classified.append([])
            for j in range(number_classes):
                # indices for evaluation_i and attack_true_labels
                classified_i = np.where(attack_test_labels[evaluation_i] == j)

                # Write evaluation indices
                eval_c = evaluation_i[classified_i]
                eval_indices_target_classified[i].append(list(eval_c))

                # Write true labels
                true_labels_classified = attack_true_labels[classified_i]
                true_labels_target_classified[i].append(list(true_labels_classified))

        target_model_indices = {
            "indices_per_target": eval_indices_target_classified,
            "true_attack_labels_per_target": true_labels_target_classified,
        }

        return target_model_indices

    @staticmethod
    def _get_target_predictions(target_models, target_data, attack_test_data):
        """
        Get classified target model predictions for every target model.

        Parameters
        ----------
        target_models : iterable
            List of target models to attack.
        target_data : dict
            Dictionary storing the classified indices and labels. Indices are in the
            evaluation data slice. The indices range starts with 0 and ends with the
            length of the evaluation data slice.

            * indices_per_target (list): Classified indices of the evaluation dataset
              per target model and per class. Shape: (target_models, classes)
            * true_attack_labels_per_target (list): The true labels for the attack
              evaluation per target model and per class. Shape: (target_models, classes)

        attack_test_data : numpy.ndarray
            Array of evaluation data.

        Returns
        -------
        list
            List of shape (target_models, classes) containing the classified
            predictions per target model and data class.
        """
        pred_list = []
        for i, target_model in enumerate(target_models):
            pred_list.append([])
            for j in range(len(target_data["indices_per_target"][i])):
                class_indices = target_data["indices_per_target"][i][j]
                pred_list[i].append(
                    target_model.predict(attack_test_data[class_indices])
                )

        return pred_list

    @staticmethod
    def _attack_model_evaluation(
        attack_models,
        target_predictions,
        target_data,
    ):
        """
        Evaluate attack models.

        Parameters
        ----------
        attack_models : iterable
            List of trained attack models to evaluate.
        target_predictions : list
            List of prediction vectors per target model and attack model.
        target_data : dict
            Dictionary storing the classified indices and labels of target training and
            evaluation records.

        Returns
        -------
        dict
            Dictionary storing the attack model results. A list "per attack model and
            target model" has the shape (attack model, target model) -> First index
            specifies the attack model, the second index the target model.

            * eval_accuracy_list (numpy.ndarray): Evaluation accuracy on test data per
              attack model and target model.
            * precision_list (numpy.ndarray): Attack precision per attack model and
              target model.
            * recall_list (numpy.ndarray): Attack recall per attack model and target
              model.
            * tp_list (numpy.ndarray): True positives per attack model and target model.
            * fp_list (numpy.ndarray): False positives per attack model and target
              model.
            * fn_list (numpy.ndarray): False negatives per attack model and target
              model.
            * tn_list (numpy.ndarray): True negatives per attack model and target model.
            * eval_accuracy (numpy.ndarray): Evaluation accuracy averaged over all
              attack models per target model.
            * precision (numpy.ndarray): Attack precision averaged over all attack
              models per target model.
            * recall (numpy.ndarray): Attack recall averaged over all attack models per
              target model.
            * overall_eval_accuracy (float): Evaluation accuracy averaged over all
              attack models and target models.
            * overall_precision (float): Attack precision averaged over all attack
              models and target models.
            * overall_recall (float): Attack recall averaged over all attack models and
              target models.
        """
        # Preallocate result matrices
        target_model_number = len(target_data["indices_per_target"])
        tn_list = np.empty((len(attack_models), target_model_number), dtype=np.int_)
        tp_list = np.empty((len(attack_models), target_model_number), dtype=np.int_)
        fn_list = np.empty((len(attack_models), target_model_number), dtype=np.int_)
        fp_list = np.empty((len(attack_models), target_model_number), dtype=np.int_)
        precision_list = np.empty((len(attack_models), target_model_number))
        recall_list = np.empty((len(attack_models), target_model_number))
        eval_accuracy_list = np.empty((len(attack_models), target_model_number))

        for tm in range(target_model_number):
            for am, attack_model in enumerate(attack_models):
                true_labels = np.array(
                    target_data["true_attack_labels_per_target"][tm][am]
                )

                # Get attack model prediction
                pred = attack_model.predict(target_predictions[tm][am])
                pred = pred.flatten()
                pred = pred > 0.5

                # Evaluate attack model prediction
                tn = np.count_nonzero((pred == False) & (true_labels == False))
                tp = np.count_nonzero((pred == True) & (true_labels == True))
                fn = np.count_nonzero((pred == False) & (true_labels == True))
                fp = np.count_nonzero((pred == True) & (true_labels == False))
                precision = tp / (tp + fp) if (tp + fp) else 1
                recall = tp / (fn + tp) if (fn + tp) else 0
                eval_accuracy = np.count_nonzero(pred == true_labels) / len(pred)

                # Store evaluation results to matrices
                tn_list[am][tm] = tn
                tp_list[am][tm] = tp
                fn_list[am][tm] = fn
                fp_list[am][tm] = fp
                precision_list[am][tm] = precision
                recall_list[am][tm] = recall
                eval_accuracy_list[am][tm] = eval_accuracy

        # Average over all attack models per target model
        precision = np.average(precision_list, axis=0)
        recall = np.average(recall_list, axis=0)
        eval_accuracy = np.average(eval_accuracy_list, axis=0)

        # Average over all attack models and target models
        precision_all = np.average(precision)
        recall_all = np.average(recall)
        eval_accuracy_all = np.average(eval_accuracy)

        attack_model_results = {
            "tn_list": tn_list,
            "tp_list": tp_list,
            "fn_list": fn_list,
            "fp_list": fp_list,
            "precision_list": precision_list,
            "recall_list": recall_list,
            "eval_accuracy_list": eval_accuracy_list,
            "eval_accuracy": eval_accuracy,
            "precision": precision,
            "recall": recall,
            "overall_eval_accuracy": eval_accuracy_all,
            "overall_precision": precision_all,
            "overall_recall": recall_all,
        }

        return attack_model_results

    @staticmethod
    def _shadow_model_evaluation(
        shadow_models,
        shadow_data_indices,
        shadow_data,
        shadow_labels,
    ):
        """
        Evaluate shadow models.

        Parameters
        ----------
        shadow_models : iterable
            List of trained shadow models to evaluate.
        shadow_data_indices : numpy.ndarray
            Array of training and evaluation indices for the shadow models
        shadow_data : numpy.ndarray
            Training and evaluation data for the shadow models.
        shadow_labels : numpy.ndarray
            Training and evaluation labels for the shadow models.

        Returns
        -------
        dict
            Dictionary storing the shadow model results.

            * shadow_train_accuracy_list (list): Accuracy on train data per shadow
              model.
            * shadow_test_accuracy_list (list): Accuracy on test data per shadow model.
            * shadow_train_accuracy (float): Accuracy on train records averaged over all
              shadow models.
            * shadow_test_accuracy (float): Accuracy on test records averaged over all
              shadow models.
        """

        train_accuracy_list = []
        test_accuracy_list = []
        train_accuracy_all = []
        test_accuracy_all = []

        for i, shadow_model in enumerate(shadow_models):
            true_train_labels = shadow_labels[shadow_data_indices[i][0]]
            true_test_labels = shadow_labels[shadow_data_indices[i][1]]

            train_pred = shadow_model.predict(shadow_data[shadow_data_indices[i][0]])
            test_pred = shadow_model.predict(shadow_data[shadow_data_indices[i][1]])

            train_pred = np.argmax(train_pred, axis=1)
            test_pred = np.argmax(test_pred, axis=1)

            train_accuracy_list.append(
                np.count_nonzero(train_pred == true_train_labels) / len(train_pred)
            )
            test_accuracy_list.append(
                np.count_nonzero(test_pred == true_test_labels) / len(test_pred)
            )

        train_accuracy_all.append(sum(train_accuracy_list) / len(train_accuracy_list))
        test_accuracy_all.append(sum(test_accuracy_list) / len(test_accuracy_list))

        shadow_model_results = {
            "shadow_train_accuracy_list": train_accuracy_list,
            "shadow_test_accuracy_list": test_accuracy_list,
            "shadow_train_accuracy": train_accuracy_all,
            "shadow_test_accuracy": test_accuracy_all,
        }
        return shadow_model_results

    @staticmethod
    def _create_shadow_model_datasets(
        origin_dataset_size, number_shadow_models, shadow_train_size, seed=None
    ):
        """
        Create datasets (containing training and evaluating data) for the shadow models.

        Parameters
        ----------
        origin_dataset_size : int
            Size of the full origin dataset.
        number_shadow_models : int
            Amount of shadow models to be trained.
        shadow_train_size : int
            Size of each shadow model's training set. The corresponding evaluation sets
            will have the same size.
        seed : int
            Explict seed for the distribution. Can be used to achieve deterministic
            behavior. If None, no explict seed will be used.

        Returns
        -------
        numpy.ndarray
            Array of shape (n, 2, m) containing m training records and m evaluation
            records for n shadow models.
        """
        rng = np.random.default_rng(seed)
        shadow_datasets = np.empty(
            (number_shadow_models, 2, shadow_train_size), dtype=np.uint32
        )
        for i in range(number_shadow_models):
            choice = rng.choice(
                origin_dataset_size, size=shadow_train_size * 2, replace=False
            )
            # Split to training and evaluation set (no overlap)
            shadow_datasets[i] = np.split(choice, 2)

        return shadow_datasets

    @staticmethod
    def _train_shadow_models(
        create_compile_shadow_model,
        shadow_data_indices,
        train_data,
        train_labels,
        epochs,
        batch_size,
    ):
        """
        Train shadow models which are based on the given create model function.

        Parameters
        ----------
        create_compile_shadow_model : function
            Return compiled TensorFlow Keras model for the shadow models.
        shadow_data_indices : numpy.ndarray
            Provides dataset mappings for the shadow models.
        train_data : numpy.ndarray
            Training data for the shadow models.
        train_labels : numpy.ndarray
            Training labels for the shadow models.
        epochs : int
            Number of training epochs for each shadow model.
        batch_size : int
            Size of mini batches used during the training.

        Returns
        -------
        list
            Array of the trained shadow models.
        """
        shadow_models = []
        number_shadow_models = len(shadow_data_indices)
        for i in range(number_shadow_models):
            logger.info(
                f"Progress: Train shadow model ({i + 1}/{number_shadow_models})."
            )
            shadow_model = create_compile_shadow_model()
            shadow_model.fit(
                train_data[shadow_data_indices[i][0]],
                train_labels[shadow_data_indices[i][0]],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            shadow_models.append(shadow_model)

        return shadow_models

    @staticmethod
    def _generate_attack_dataset(
        shadow_models, shadow_data_indices, number_classes, shadow_data, shadow_labels
    ):
        """
        Generate training data for the attack models.

        Parameters
        ----------
        shadow_models : list
            List of trained shadow models.
        shadow_data_indices : numpy.ndarray
            Array with train data and evaluation data for every shadow model.
        number_classes : int
            Number of different classes in the dataset.
        shadow_data : numpy.ndarray
            Array of data records for the shadow models.
        shadow_labels : numpy.ndarray
            Array of label records for the shadow models.

        Returns
        -------
        list
            List of dictionaries for every class containing data indices, prediction
            vectors and attack model labels (in or out).
        """
        shadow_data_size = len(shadow_data_indices[0][0]) * 2
        attack_dataset_size = shadow_data_size * len(shadow_models)
        # Preallocate
        attack_train_data = {
            "indices": np.empty(attack_dataset_size, dtype=np.uint32),
            "prediction_vectors": np.empty(
                (attack_dataset_size, number_classes), dtype=np.float32
            ),
            "attack_labels": np.empty(attack_dataset_size, dtype=np.bool_),
        }

        # Populate
        logger.info(f"Get prediction of shadow models.")
        in_array = np.full(int(shadow_data_size / 2), True, dtype=np.bool_)
        out_array = np.full(int(shadow_data_size / 2), False, dtype=np.bool_)
        for i, shadow_model in enumerate(shadow_models):
            # Make predictions
            logger.debug(
                f"Get prediction of shadow model ({i+1}/{len(shadow_models)})."
            )
            in_prediction = shadow_model.predict(shadow_data[shadow_data_indices[i][0]])
            out_prediction = shadow_model.predict(
                shadow_data[shadow_data_indices[i][1]]
            )
            # Populate unclassified attack dataset
            start_index = i * shadow_data_size
            end_index = start_index + shadow_data_size
            attack_train_data["indices"][start_index:end_index] = np.concatenate(
                (shadow_data_indices[i][0], shadow_data_indices[i][1])
            )
            attack_train_data["prediction_vectors"][
                start_index:end_index
            ] = np.concatenate((in_prediction, out_prediction))
            attack_train_data["attack_labels"][start_index:end_index] = np.concatenate(
                (in_array, out_array)
            )

        # Classify attack data
        logger.debug(f"Classify attack train data.")
        attack_train_data_classified = []
        for i in range(number_classes):
            indices = np.where(shadow_labels[attack_train_data["indices"]] == i)
            attack_train_data_classified.append(
                {
                    "indices": attack_train_data["indices"][indices],
                    "prediction_vectors": attack_train_data["prediction_vectors"][
                        indices
                    ],
                    "attack_labels": attack_train_data["attack_labels"][indices],
                }
            )

        return attack_train_data_classified

    @staticmethod
    def _train_attack_models(
        create_compile_attack_model,
        attack_datasets,
        epochs,
        batch_size,
    ):
        """
        Train attack models which are based on the given create model function.

        Parameters
        ----------
        create_compile_attack_model : function
            Return compiled TensorFlow Keras model for the attack models.
        attack_datasets : list
            List of generated datasets for the attack models.
        epochs : int
            Number of training epochs for each shadow model.
        batch_size : int
            Size of mini batches used during the training.

        Returns
        -------
        list
            Array of the trained attack models.
        """
        attack_models = []
        number_attack_models = len(attack_datasets)
        for i in range(number_attack_models):
            logger.info(
                f"Progress: Train attack model ({i + 1}/{number_attack_models})."
            )
            shadow_model = create_compile_attack_model()
            shadow_model.fit(
                attack_datasets[i]["prediction_vectors"],
                attack_datasets[i]["attack_labels"],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
            )
            attack_models.append(shadow_model)

        return attack_models

    def create_attack_report(self, save_path="mia_report", pdf=False):
        """
        Create an attack report just for the given attack instantiation.

        Parameters
        ----------
        save_path : str
            Path to save the tex, pdf and asset files of the attack report.
        pdf : bool
            If set, generate pdf out of latex file.
        """

        # Create directory structure for the attack report, including the figure
        # directory for the figures of the results subsubsection.
        os.makedirs(save_path + "/fig", exist_ok=True)

        self.create_attack_section(save_path=save_path)
        report.report_generator(save_path, [self.report_section], pdf)

    def create_attack_section(self, save_path):
        """
        Create an attack section for the given attack instantiation.

        Parameters
        ----------
        save_path :
            Path to save the tex, pdf and asset files of the attack report.
        """
        self._report_attack_configuration()
        self._report_attack_results(save_path)

    def _report_attack_configuration(self):
        """Create subsubsection about the attack and data configuration."""
        # Create tables for attack parameters and the data configuration.
        ap = self.attack_pars
        dc = self.data_conf
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
                        ["Number of shadow models", ap["number_shadow_models"]]
                    )
                    tab_ap.add_hline()
                    tab_ap.add_row(
                        [
                            "Shadow training set size",
                            ap["shadow_training_set_size"],
                        ]
                    )
                    tab_ap.add_hline()
                    tab_ap.add_row(["Shadow epochs", ap["shadow_epochs"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(["Shadow batch size", ap["shadow_batch_size"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(["Attack epochs", ap["attack_epochs"]])
                    tab_ap.add_hline()
                    tab_ap.add_row(["Attack batch size", ap["attack_batch_size"]])
                    tab_ap.add_hline()
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "table",
                        extra_arguments="Attack parameters",
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
                            NoEscape("Samples used to train shadow models ($S$)"),
                            len(dc["shadow_indices"]),
                        ]
                    )
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        [
                            NoEscape("Samples used to train target models ($T$)"),
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
                            NoEscape("Size of $S \cap T$"),
                            len(set(dc["shadow_indices"]) & set(dc["target_indices"])),
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
        """
        Create subsubsection describing the most important results of the attack.

        Parameters
        ----------
        save_path :
            Path to save the tex, pdf and asset files of the attack report.

        This subsection contains results only for the first target model.
        """
        tm = 0  # Specify target model
        self.report_section.append(Subsubsection("Attack Results"))
        res = self.attack_results

        # ECDF graph (like in paper)
        precision_sorted = np.sort(res["precision_list"], axis=0)[:, tm]
        recall_sorted = np.sort(res["recall_list"], axis=0)[:, tm]
        py = np.arange(1, len(precision_sorted) + 1) / len(precision_sorted)
        ry = np.arange(1, len(recall_sorted) + 1) / len(recall_sorted)

        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Cumulative Fraction of Classes")
        ax.plot(precision_sorted, py, "k-", label="Precision")
        ax.plot(recall_sorted, ry, "k--", label="Recall")
        ax.legend()

        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        fig.savefig(save_path + f"fig/{alias_no_spaces}-ecdf.pdf", bbox_inches="tight")
        plt.close(fig)

        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                self.report_section.append(
                    Command(
                        "includegraphics",
                        NoEscape(f"fig/{alias_no_spaces}-ecdf.pdf"),
                        "width=8cm",
                    )
                )
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "figure",
                        extra_arguments="Empirical CDF",
                    )
                )

            tp_row = []
            fp_row = []
            tn_row = []
            fn_row = []
            class_row = []
            precision_row = []
            accuracy_row = []
            recall_row = []

            # Average column
            class_row.append(f"0-{self.attack_pars['number_classes']-1}")
            tp_row.append(
                np.round(np.sum(res["tp_list"], axis=0)[tm] / len(res["tp_list"]), 2)
            )
            fp_row.append(
                np.round(np.sum(res["fp_list"], axis=0)[tm] / len(res["fp_list"]), 2)
            )
            tn_row.append(
                np.round(np.sum(res["tn_list"], axis=0)[tm] / len(res["tn_list"]), 2)
            )
            fn_row.append(
                np.round(np.sum(res["fn_list"], axis=0)[tm] / len(res["fn_list"]), 2)
            )
            accuracy_row.append(
                np.round(
                    np.sum(res["eval_accuracy_list"], axis=0)[tm]
                    / len(res["eval_accuracy_list"]),
                    3,
                )
            )
            precision_row.append(
                np.round(
                    np.sum(res["precision_list"], axis=0)[tm]
                    / len(res["precision_list"]),
                    3,
                )
            )
            recall_row.append(
                np.round(
                    np.sum(res["recall_list"], axis=0)[tm] / len(res["recall_list"]), 3
                )
            )

            # Maximum accuracy class
            max_class = np.argmax(res["eval_accuracy_list"], axis=0)[tm]
            class_row.append(max_class)
            tp_row.append(res["tp_list"][max_class][tm])
            fp_row.append(res["fp_list"][max_class][tm])
            tn_row.append(res["tn_list"][max_class][tm])
            fn_row.append(res["fn_list"][max_class][tm])
            accuracy_row.append(np.round(res["eval_accuracy_list"][max_class][tm], 3))
            precision_row.append(np.round(res["precision_list"][max_class][tm], 3))
            recall_row.append(np.round(res["recall_list"][max_class][tm], 3))

            # Minimum accuracy class
            min_class = np.argmin(res["eval_accuracy_list"], axis=0)[tm]
            class_row.append(min_class)
            tp_row.append(res["tp_list"][min_class][tm])
            fp_row.append(res["fp_list"][min_class][tm])
            tn_row.append(res["tn_list"][min_class][tm])
            fn_row.append(res["fn_list"][min_class][tm])
            accuracy_row.append(np.round(res["eval_accuracy_list"][min_class][tm], 3))
            precision_row.append(np.round(res["precision_list"][min_class][tm], 3))
            recall_row.append(np.round(res["recall_list"][min_class][tm], 3))

            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                with self.report_section.create(Tabular("|l|c|c|c|")) as result_tab:
                    result_tab.add_hline()
                    result_tab.add_row(
                        list(
                            map(
                                bold,
                                [
                                    "",
                                    "Average",
                                    "max. Acc.",
                                    "min. Acc.",
                                ],
                            )
                        )
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["Class", class_row[0], class_row[1], class_row[2]]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["True Positives", tp_row[0], tp_row[1], tp_row[2]]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["False Positives", fp_row[0], fp_row[1], fp_row[2]]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["True Negatives", tn_row[0], tn_row[1], tn_row[2]]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["False Negatives", fn_row[0], fn_row[1], fn_row[2]]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["Accuracy", accuracy_row[0], accuracy_row[1], accuracy_row[2]]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        [
                            "Precision",
                            precision_row[0],
                            precision_row[1],
                            precision_row[2],
                        ]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["Recall", recall_row[0], recall_row[1], recall_row[2]]
                    )
                    result_tab.add_hline()
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command("captionof", "table", extra_arguments="Attack Summary")
                )

        ap = self.attack_pars

        # Histograms
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(12, 3))
        ax0.hist(res["eval_accuracy_list"][:, tm], edgecolor="black")
        ax1.hist(res["precision_list"][:, tm], edgecolor="black")
        ax2.hist(res["recall_list"][:, tm], edgecolor="black")
        ax0.set_xlabel("Accuracy")
        ax1.set_xlabel("Precision")
        ax2.set_xlabel("Recall")
        ax0.set_ylabel("Number of Classes")
        ax0.tick_params(axis="x", labelrotation=45)
        ax1.tick_params(axis="x", labelrotation=45)
        ax2.tick_params(axis="x", labelrotation=45)
        ax0.set_axisbelow(True)
        ax1.set_axisbelow(True)
        ax2.set_axisbelow(True)

        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        fig.savefig(
            save_path + f"/fig/{alias_no_spaces}-accuracy_precision_recall.pdf", bbox_inches="tight"
        )
        plt.close(fig)

        with self.report_section.create(Figure(position="H")) as fig:
            fig.add_image(
                f"fig/{alias_no_spaces}-accuracy_precision_recall.pdf", width=NoEscape(r"\textwidth")
            )
            self.report_section.append(Command("captionsetup", "labelformat=empty"))
            self.report_section.append(
                Command(
                    "captionof",
                    "figure",
                    extra_arguments="Accuracy, precision and recall histogram for all "
                    "attack models",
                )
            )
