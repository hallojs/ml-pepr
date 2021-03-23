"""Membership inference attack against machine learning models."""

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

from pepr import attack, report

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

plt.style.use("seaborn-white")


class Mia(attack.Attack):
    """
    Membership Inference Attacks (MIA) Against Machine Learning Models

    Attack-Steps:
    TODO: doc - describe attack

    Parameters
    ----------
    TODO: doc - finish param list
    attack_alias : str
            Alias for a specific instantiation of the mia.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * number_classes (int): Number of different classes the target model predicts.
        * number_shadow_models (int): Number of shadow models to be trained.
        * shadow_training_set_size (int): Size of the trainings set for each
          shadow model. The corresponding evaluation sets will have the same size.
        * create_compile_model (function): Function that returns a compiled
          TensorFlow model (typically identical to the target model) used in the
          training of the shadow models.
        * create_compile_attack_model (function): Function that returns a compiled
          TensorFlow model used for the attack models.
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
        setting.
    target_models : iterable
        List of target models which should be tested.
    attack_results : dict

        * precision_list (list): Attack precision per attack model and target model.
        * recall_list (list): Attack recall per attack model and target model.
        * tp_list (list): True positives per attack model and target model.
        * fp_list (list): False positives per attack model and target model.
        * fn_list (list): False negatives per attack model and target model.
        * tn_list (list): True negatives per attack model and target model.
        * precision (float): Attack precision averaged over all attack models per target
          model.
        * recall (float): Attack recall averaged over all attack models per target
          model.
        * test_accuracy (float): Attack test accuracy averaged over all attack models.
        * overall_precision (float): Attack precision averaged over all target models.
        * overall_recall (float): Attack recall averaged over all target models.
        * test_accuracy_list (list): Accuracy on evaluation records per attack model and
          target model.
        * shadow_train_accuracy_list (list): Accuracy on training records per shadow
          model and target model.
        * shadow_test_accuracy_list (list): Accuracy on evaluation records per shadow
          model and target model.
        * shadow_train_accuracy (float): Accuracy on train records averaged over all
          shadow models per target model.
        * shadow_test_accuracy (float): Accuracy on evaluation records averaged over all
          shadow models per target model.
        * target_train_accuracy_list (list): Accuracy on train records per target model.
        * target_test_accuracy_list (list): Accuracy on evaluation records per target
          model.
        * target_train_accuracy (float): Accuracy on train records averaged over all
          target models.
        * target_test_accuracy (float): Accuracy on evaluation records averaged over all
          target models.

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

        # Setup shadow model datasets:
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

        # Train shadow models
        if load and "shadow_models" in load_pars.keys():
            paths = load_pars["shadow_models"]
            shadow_models = []
            for path in paths:
                logger.info(f"Load pre-trained shadow models: {path}.")
                shadow_models.append(models.load_model(path))
        else:
            logger.info("Train shadow models.")
            shadow_models = Mia._train_shadow_models(
                self.attack_pars["create_compile_model"],
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

        # Attack model dataset generation
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

        # Train attack models
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

        # Evaluate attack models
        logger.info("Evaluate attack models.")
        # -- Target prediction + true label
        # TODO: Support more than one target model
        target_prediction = self.target_models[0].predict(attack_test_data)
        attack_test_bool = np.concatenate(
            (
                np.full(
                    len(self.data_conf["record_indices_per_target"][0]),
                    True,
                    dtype=np.bool_,
                ),
                np.full(
                    len(attack_test_data)
                    - len(self.data_conf["record_indices_per_target"][0]),
                    False,
                    dtype=np.bool_,
                ),
            )
        )

        # -- Classifiy attack evaluation dataset (prediction vectors)
        attack_test_indices_classified = []
        for i in range(self.attack_pars["number_classes"]):
            attack_test_indices_classified.append(np.where(attack_test_labels == i))

        # -- Evaluate
        logger.info("Calculate attack model summary.")
        self.attack_results = Mia._attack_model_evaluation(
            attack_models,
            target_prediction,
            attack_test_bool,
            attack_test_indices_classified,
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
        logger.info("Calculate target model summary.")
        not_target_test_indices = np.searchsorted(
            self.data_conf["evaluation_indices"],
            np.array(self.data_conf["target_indices"]).flatten(),
        )
        self.attack_results.update(
            Mia._target_model_evaluation(
                self.target_models,
                self.data[self.data_conf["target_indices"]],
                self.labels[self.data_conf["target_indices"]],
                np.delete(attack_test_data, not_target_test_indices, axis=0),
                np.delete(attack_test_labels, not_target_test_indices, axis=0),
                self.data_conf["record_indices_per_target"],
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
            f"\n################# Target and Shadow Results ################"
            f"\n"
            f"\n{'Target Model:':<30}"
            + _list_to_formatted_string(range(len(self.target_models)))
            + f"\n{'Training Accuracy:':<30}"
            + _list_to_formatted_string(
                self.attack_results["target_train_accuracy_list"]
            )
            + f"\n{'Evaluation Accuracy:':<30}"
            + _list_to_formatted_string(
                self.attack_results["target_test_accuracy_list"]
            )
            + f"\n{'Average Training Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["target_train_accuracy"])
            + f"\n{'Average Evaluation Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["target_test_accuracy"])
            + f"\n"
            + f"\n{'Shadow Model:':<30}"
            + _list_to_formatted_string(range(len(shadow_models)))
            + f"\n{'Training Accuracy:':<30}"
            + _list_to_formatted_string(
                self.attack_results["shadow_train_accuracy_list"]
            )
            + f"\n{'Evaluation Accuracy:':<30}"
            + _list_to_formatted_string(
                self.attack_results["shadow_test_accuracy_list"]
            )
            + f"\n{'Average Training Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["shadow_train_accuracy"])
            + f"\n{'Average Evaluation Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["shadow_test_accuracy"])
            + f"\n"
            f"\n###################### Attack Results ######################"
            f"\n"
            f"\n{'Attack Model:':<30}"
            + _list_to_formatted_string(range(len(attack_models)))
            + f"\n{'True Positives:':<30}"
            + _list_to_formatted_string(self.attack_results["tp_list"])
            + f"\n{'False Positives:':<30}"
            + _list_to_formatted_string(self.attack_results["fp_list"])
            + f"\n{'True Negatives:':<30}"
            + _list_to_formatted_string(self.attack_results["tn_list"])
            + f"\n{'False Negatives:':<30}"
            + _list_to_formatted_string(self.attack_results["fn_list"])
            + f"\n{'Test Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["test_accuracy_list"])
            + f"\n{'Precision:':<30}"
            + _list_to_formatted_string(self.attack_results["precision_list"])
            + f"\n{'Recall:':<30}"
            + _list_to_formatted_string(self.attack_results["recall_list"])
            + f"\n"
            f"\n{'Average Test Accuracy:':<30}"
            + _list_to_formatted_string(self.attack_results["test_accuracy"])
            + f"\n{'Average Precision:':<30}"
            + _list_to_formatted_string(self.attack_results["precision"])
            + f"\n{'Average Recall:':<30}"
            + _list_to_formatted_string(self.attack_results["recall"])
        )

    @staticmethod
    def _attack_model_evaluation(
        attack_models,
        attack_test_data,
        attack_test_true_labels,
        attack_test_indices_classified,
    ):
        """
        Evaluate attack models.

        Parameters
        ----------
        attack_models : iterable
            List of trained attack models to evaluate.
        attack_test_data : numpy.ndarray
            Array of prediction vectors of the target model.
        attack_test_true_labels : numpy.ndarray
            Array of the true classification of the prediction vectors.
        attack_test_indices_classified : iterable
            Array of indices mapping the attack evaluation data to classes debending on
            the true label.

        Returns
        -------
        dict
            Dictionary storing the attack model results.

            * precision_list (list): Attack precision per attack model.
            * recall_list (list): Attack recall per attack model.
            * tp_list (list): True positives per attack model.
            * fp_list (list): False positives per attack model.
            * fn_list (list): False negatives per attack model.
            * tn_list (list): True negatives per attack model.
            * precision (float): Attack precision averaged over all attack models.
            * recall (float): Attack recall averaged over all attack models.
            * test_accuracy_list (list): Accuracy on test data per attack model.
            * test_accuracy (float): Attack test accuracy averaged over all attack
              models.
        """
        tn_list = []
        tp_list = []
        fn_list = []
        fp_list = []
        precision_list = []
        recall_list = []
        test_accuracy_list = []
        precision_all = []
        recall_all = []
        accuracy_all = []

        for i, attack_model in enumerate(attack_models):
            true_labels = attack_test_true_labels[attack_test_indices_classified[i]]
            pred = attack_model.predict(
                attack_test_data[attack_test_indices_classified[i]]
            )
            pred = np.argmax(pred, axis=1)

            tn = np.count_nonzero((pred == False) & (true_labels == False))
            tp = np.count_nonzero((pred == True) & (true_labels == True))
            fn = np.count_nonzero((pred == False) & (true_labels == True))
            fp = np.count_nonzero((pred == True) & (true_labels == False))
            precision = tp / (tp + fp) if (tp + fp) else 1
            recall = tp / (fn + tp) if (fn + tp) else 0
            test_accuracy = np.count_nonzero(pred == true_labels) / len(pred)

            tn_list.append(tn)
            tp_list.append(tp)
            fn_list.append(fn)
            fp_list.append(fp)
            precision_list.append(precision)
            recall_list.append(recall)
            test_accuracy_list.append(test_accuracy)

        precision_all.append(sum(precision_list) / len(precision_list))
        recall_all.append(sum(recall_list) / len(recall_list))
        accuracy_all.append(sum(test_accuracy_list) / len(test_accuracy_list))

        attack_model_results = {
            "tn_list": tn_list,
            "tp_list": tp_list,
            "fn_list": fn_list,
            "fp_list": fp_list,
            "precision_list": precision_list,
            "recall_list": recall_list,
            "test_accuracy_list": test_accuracy_list,
            "test_accuracy": accuracy_all,
            "precision": precision_all,
            "recall": recall_all,
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
            * shadow_test_accuracy_list (list): Accuracy on evaluation data per shadow
              model.
            * shadow_train_accuracy (float): Accuracy on train records averaged over all
              shadow models.
            * shadow_test_accuracy (float): Accuracy on evaluation records averaged over
              all shadow models.
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
    def _target_model_evaluation(
        target_models,
        train_data,
        train_labels,
        test_data,
        test_labels,
        indices_per_target,
    ):
        """
        Evaluate target models.

        Parameters
        ----------
        target_models : list
            List of target models to evaluate.
        train_data : numpy.ndarray
            Training data for the target models.
        train_labels : numpy.ndarray
            Training labels for the target models.
        test_data : numpy.ndarray
            Evaluation data for the target models.
        test_labels : numpy.ndarray
            Evaluation labels for the target models.

        Returns
        -------
        dict
            Dictionary storing the shadow model results.

            * target_train_accuracy_list (list): Accuracy on train data per target
              model.
            * target_test_accuracy_list (list): Accuracy on evaluation data per target
              model.
            * target_train_accuracy (float): Accuracy on train records averaged over all
              target models.
            * target_test_accuracy (float): Accuracy on evaluation records averaged over
              all target models.
        """
        train_accuracy_list = []
        test_accuracy_list = []
        train_accuracy_all = []
        test_accuracy_all = []

        for i, target_model in enumerate(target_models):
            train_pred = target_model.predict(train_data[indices_per_target[i]])
            test_pred = target_model.predict(test_data[indices_per_target[i]])

            train_pred = np.argmax(train_pred, axis=1)
            test_pred = np.argmax(test_pred, axis=1)

            train_accuracy_list.append(
                np.count_nonzero(train_pred == train_labels[indices_per_target[i]])
                / len(train_pred)
            )
            test_accuracy_list.append(
                np.count_nonzero(test_pred == test_labels[indices_per_target[i]])
                / len(test_pred)
            )

        train_accuracy_all.append(sum(train_accuracy_list) / len(train_accuracy_list))
        test_accuracy_all.append(sum(test_accuracy_list) / len(test_accuracy_list))

        target_model_results = {
            "target_train_accuracy_list": train_accuracy_list,
            "target_test_accuracy_list": test_accuracy_list,
            "target_train_accuracy": train_accuracy_all,
            "target_test_accuracy": test_accuracy_all,
        }
        return target_model_results

    def create_attack_report(self, save_path):
        """Create an attack report just for the given attack instantiation.

        Parameters
        ----------
        save_path : str
            Path to save the tex and pdf file of the attack report.
        """
        pass

    def create_attack_section(self, save_path):
        """Create an attack section for the given attack instantiation.

        Parameters
        ----------
        save_path :
            Path to save the report assets like figures.
        """
        pass

    @staticmethod
    def _create_shadow_model_datasets(
        origin_dataset_size, number_shadow_models, shadow_train_size, seed=None
    ):
        """
        Create datasets (for training and evaluating) for the shadow models.

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
        create_compile_model,
        shadow_data_indices,
        train_data,
        train_labels,
        epochs,
        batch_size,
    ):
        """
        Trains shadow models based on the model which create_compile_model returns.

        Parameters
        ----------
        create_compile_model : function
            Return compiled TensorFlow Keras model for the shadow models.
        shadow_data_indices : numpy.ndarray
            Provides dataset mappings for the shadow models.
        train_data : numpy.ndarray
            Training data for the shadow models.
        train_labels : numpy.ndarray
            Traininf labels for the shadow models.
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
            shadow_model = create_compile_model()
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
            Array with train data and evaluation data dor every shadow model.
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
            vectors and attack model labels (in or out)
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
        in_array = np.full(int(shadow_data_size / 2), True, dtype=np.bool_)
        out_array = np.full(int(shadow_data_size / 2), False, dtype=np.bool_)
        for i, shadow_model in enumerate(shadow_models):
            # Make predictions
            logger.info(f"Get prediction of shadow model ({i+1}/{len(shadow_models)}).")
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
        Trains attack models based on the model which create_compile_attack_model
        returns.

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
