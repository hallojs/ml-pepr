"""Direct mia on a single target cnn.

Implementation of the direct mia from Long, Yunhui and Bindschaedler, Vincent and Wang,
Lei and Bu, Diyue and Wang, Xiaofeng and Tang, Haixu and Gunter, Carl A and Chen, Kai
(2018). Understanding membership inferences on well-generalized learning models. arXiv
preprint arXiv:1802.04889.
"""
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

# for the empirical cdf
from statsmodels.distributions.empirical_distribution import ECDF

# for the interpolation
from scipy.interpolate import pchip
from tensorflow.keras.models import Model
from tensorflow.keras import models

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DirectMia:
    """Whole functionality needed for the direct mia."""

    @staticmethod
    def assign_records_to_reference_models(
        number_reference_models, background_knowledge_size, training_set_size, path
    ):
        """Create training datasets for the reference models.

        Parameters
        ----------
        number_reference_models : int
            Number of reference models to be trained
        background_knowledge_size : int
            Size of the background knowledge of the attacker.
        training_set_size : int
            Number of samples used to train each reference model.
        path : str
            Path for saving the reference models.

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

        np.save(path, records_per_reference_model)

        return records_per_reference_model

    @staticmethod
    def train_reference_models(
        create_compile_model,
        records_per_reference_model,
        train_data,
        train_labels,
        epochs,
        batch_size,
        save_path,
    ):
        """Train reference models based on a compiled TensorFlow Keras model.

        Parameters
        ----------
        create_compile_model : function
            Return compiled TensorFlow Keras model, used to train the reference
            models.
        records_per_reference_model : np.ndarray
            Describes which record is used to train which model.
        train_data : numpy.ndarray
            Training data for the reference models.
        train_labels : numpy.ndarray
            Training labels (one-hot encoding) for the reference models.
        epochs : int
            Number of training epochs for each reference model.
        batch_size : int
            Size of mini batches used during the training.
        save_path : str
            Path to save the reference models.

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
            tmp_model.save(save_path + str(i))
            reference_models.append(tmp_model)

        return np.asarray(reference_models)

    @staticmethod
    def load_models(path, number_models):
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
    def gen_intermediate_models(models, layer_number):
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
        if type(models) != np.ndarray:
            layer = models.layers[layer_number]._name
            layer_output = models.get_layer(layer).output
            intermediate_model = Model(inputs=models.input, outputs=layer_output)
            return intermediate_model

        intermediate_models = np.array([])
        for i, model in enumerate(models):
            layer = model.layers[layer_number]._name
            layer_output = model.get_layer(layer).output
            # noinspection PyTypeChecker
            intermediate_models = np.append(
                intermediate_models, Model(inputs=model.input, outputs=layer_output)
            )
        logger.info(f"Generated {len(models)} intermediate models.")
        return intermediate_models

    # TODO: Add more precise description about the extraction process.

    @staticmethod
    def extract_high_level_features(intermediate_models, data, path):
        """Extract the high level features from the intermediate models.

        For details see paper section 4.3.

        Parameters
        ----------
        intermediate_models : numpy.ndarray
            Array of intermediate models. It is also possible to pass a single
            model.
        data : numpy.ndarray
            Data from which the high-level features should be extracted.
        path : str
            Path to save the high-level-features.

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
        np.save(path, feature_vecs)
        logger.info(
            f"Extracted high-level-features from {len(intermediate_models)} "
            f"intermediate models."
        )
        return feature_vecs

    @staticmethod
    def calc_pairwise_distances(
        features_target, features_reference, path, metric, n_jobs=1
    ):
        """Calculate pairwise distances between given features.

        Parameters
        ----------
        features_target : numpy.ndarray
            First array for pairwise distances.
        features_reference : numpy.ndarray
            Second array for pairwise distances.
        path : str
            Path to save array of distances.
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
        np.save(path, distances)

        return distances

    @staticmethod
    def select_target_records(
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
        training sets. For details see section 4.3 from the paper.

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

        logger.info(f"min_distance: {np.min(distances)}")
        logger.info(f"mean n_neighbors: {np.mean(n_neighbors)}")
        logger.info(f"mean est_n_neighbors: {np.mean(est_n_neighbors)}")
        logger.info(f"number of target_records: {len(target_records)}")
        logger.info(f"target_records: {target_records}")
        logger.info(f"number of neighbors: {n_neighbors[target_records]}")

        return target_records

    @staticmethod
    def get_model_inference(idx_records, records, labels_cat, models):
        """Predict on trained models and calculate the log loss.

        Parameters
        ----------
        idx_records : numpy.ndarray
            Index array of records to predict on.
        records : numpy.ndarray
            Array of records used for prediction.
        labels_cat : numpy.ndarray
            Array of labels used to predict on (one-hot encoded).
        models : numpy.ndarray
            Array of models used for prediction. It is also possible to pass only
            one model (not in an numpy.ndarray).

        Returns
        -------
        numpy.array
            Array of log losses of predictions.
        """
        if type(models) != np.ndarray:
            models = np.array([models])

        inferences = []
        for i, model in enumerate(models):
            logger.info(f"Do inference on model {i+1}/{len(models)}.")
            predictions = model.predict(records[idx_records])
            filter_predictions = np.max(predictions * labels_cat[idx_records], axis=1)
            inferences.append(-np.log(filter_predictions))

        return np.asarray(inferences).T

    @staticmethod
    def sample_reference_losses(
        target_records, reference_inferences, print_target_information
    ):
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
        print_target_information : boolean
            If true print target information.

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

        if print_target_information:
            logging.info(f"Number of used target records: {len(used_target_records)}")
            logging.info(f"Used target records (indexes): {used_target_records}")

        return used_target_records, pchip_references, ecdf_references

    @staticmethod
    def hypothesis_test(
        cut_off_p_value, pchip_references, target_inferences, used_target_records
    ):
        """Left-tailed hypothesis test.

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


class AttackVisualisations:
    """Collection of functions to visualize the results of the attacks."""

    @staticmethod
    def plot_target_records(target_records, target_data, target_labels, input_shape):
        """Plot target records to get a understanding of our selection algorithm.

        Parameters
        ----------
        target_records : numpy.ndarray
            Selected target records which should be plotted.
        target_data : numpy.ndarray
            The data set we are attacking.
        target_labels : numpy.ndarray
            The labels to the data we are attacking. Not one-hot encoded!
        input_shape : tuple
            Shape of the samples (images).
        """
        rows = math.ceil(len(target_records) / 3)
        plt.figure(figsize=[15, rows * 4])
        for idx, target_record in enumerate(target_records):
            title = (
                "record="
                + str(target_record)
                + " label="
                + str(target_labels[target_record])
            )

            ax = plt.subplot(rows, 3, idx + 1)
            # ax = plt.subplot(rows, 6, idx + 1)
            ax.set_title(title, size=25)

            if input_shape[2] <= 1:
                re_shape = (input_shape[0], input_shape[1])
            else:
                re_shape = (input_shape[0], input_shape[1], input_shape[2])

            plt.axis("off")
            plt.imshow(target_data[target_record, :, :].reshape(re_shape))

    @staticmethod
    def plot_sampled_reference_losses(
        used_target_records, ecdf_references, pchip_references
    ):
        """Plot ecdfs and pchips of sampled log losses from the reference models.

        Plot empirical cumulative distribution functions of sampled log losses
        for each used target record from the reference models in green and the
        shape preserving cubic interpolation of this ecdfs in stashed red.

        Parameters
        ----------
        used_target_records : numpy.ndarray
            Array of used target records.
        ecdf_references : list
            List of ecdfs described above.
        pchip_references : list
            List of pchips described above.
        """
        rows = math.ceil(len(used_target_records) / 3)
        plt.figure(figsize=[15, rows * 4])
        cnt = 0
        for idx, _ in enumerate(used_target_records):
            max_x = np.max(ecdf_references[idx].x[1:])
            min_x = np.min(ecdf_references[idx].y[1:])
            x = np.linspace(min_x, max_x, 1000)

            title = "Empirical CDF of $\mathcal{D}(L)$, with $r=$" + str(
                used_target_records[idx]
            )
            plt.subplot(rows, 3, cnt + 1, title=title)
            plt.plot(
                ecdf_references[idx].x,
                ecdf_references[idx].y,
                color="green",
                linewidth=3,
                label="empirical cdf",
            )
            plt.plot(
                x,
                pchip_references[idx](x),
                color="red",
                linestyle="dotted",
                linewidth=3,
                label="pchip",
            )
            plt.legend()
            cnt += 1


class OotbDirectMia:
    """A class that allows to run the direct mia with just 3 function calls."""

    @staticmethod
    def prepare_attack(
        train_data,
        train_labels_cat,
        epochs,
        batch_size,
        number_reference_models,
        background_knowledge_size,
        training_set_size,
        save_path,
        create_compile_model,
    ):
        """Prepare reference models for the direct membership inference attack.

        1. Create mapping of records to reference models.
        2. Train the reference models.

        Parameters
        ----------
        train_data : numpy.ndarray
            Training data for the reference models.
        train_labels_cat : numpy.ndarray
            Training labels (one-hot encoding) for the reference models.
        epochs : int
            Number of training epochs for each reference model.
        batch_size : int
            Size of mini batches used during the training.
        number_reference_models : int
            Number of reference models to be trained.
        background_knowledge_size : int
            Size of the background knowledge of the attacker.
        training_set_size : int
            Size of training set of the reference models
        save_path : str
            Path to save the records to reference model mapping and the trained
            reference models.
        create_compile_model : function
            Return compiled TensorFlow Keras model, used to train the reference
            models.

        Returns
        -------
        numpy.ndarray
            Array of trained reference models
        """
        tmp_path = save_path + "/records_per_reference_model.npy"
        print("Create mapping of records to reference models: ", tmp_path)
        records_per_reference_model = DirectMia.assign_records_to_reference_models(
            number_reference_models,
            background_knowledge_size,
            training_set_size,
            tmp_path,
        )

        tmp_path = save_path + "/reference_model"
        print("Start training of the reference models: ", tmp_path)
        reference_models = DirectMia.train_reference_models(
            create_compile_model,
            records_per_reference_model,
            train_data,
            train_labels_cat,
            epochs,
            batch_size,
            tmp_path,
        )
        return reference_models

    @staticmethod
    def select_target_records(
        reference_models,
        layer_number,
        reference_train_data,
        save_path,
        attack_data,
        metric,
        training_set_size,
        background_knowledge_size,
        probability_threshold,
        neighbor_threshold,
        load_distances=False,
    ):
        """Select potential vulnerable target records.

        1. Generate intermediate models.
        2. Extract reference high-level features.
        3. Extract target high-level features.
        4. Compute pairwise distances between reference and target high-level
           features.
        5. Determine potential vulnerable target records.

        Parameters
        ----------
        reference_models : numpy.ndarray
            Array of the trained reference models.
        layer_number : int
            Number of the intermediate layer used as the new output layer in the
            intermediate models.
        reference_train_data : numpy.ndarray
            Training data for the reference models.
        save_path : str
            Path to save high-level features and the pairwise distance matrix.
        attack_data : numpy.ndarray
            Array of target samples.
        metric : str
            Metric used for the distance computations in the high-level feature
            space.
        training_set_size : int
            Number of samples used to train target model.
        background_knowledge_size : int
            Size of the background knowledge of the attacker.
        probability_threshold : float
            For details see section 4.3 from the paper.
        neighbor_threshold : float
            If distance is smaller then the neighbor threshold the record is
            selected as target record.
        load_distances : bool, optional
            If true function loads distance matrix from save_path.

        Returns
        -------
        numpy.ndarray
            Array of potential vulnerable target records.
        """
        if load_distances:
            tmp_path = (
                save_path + "/pairwise_distances_high_level_features_" + metric + ".npy"
            )
            print("Load distance matrix :", tmp_path)
            distances = np.load(tmp_path)
        else:
            print("Generate intermediate models")
            reference_intermediate_models = DirectMia.gen_intermediate_models(
                reference_models, layer_number
            )

            tmp_path = save_path + "/reference_high_level_features.npy"
            print("Extract reference high-level features: ", tmp_path)
            reference_high_level_features = DirectMia.extract_high_level_features(
                reference_intermediate_models, reference_train_data, tmp_path
            )

            tmp_path = save_path + "/target_high_level_features.npy"
            print("Extract target high-level features: ", tmp_path)
            target_high_level_features = DirectMia.extract_high_level_features(
                reference_intermediate_models, attack_data, tmp_path
            )

            tmp_path = (
                save_path + "/pairwise_distances_high_level_features_" + metric + ".npy"
            )
            print(
                "Compute pairwise distances between reference and target "
                + "high-level features: ",
                tmp_path,
            )
            distances = DirectMia.calc_pairwise_distances(
                target_high_level_features,
                reference_high_level_features,
                tmp_path,
                metric,
            )

        print("Determine potential vulnerable target records")
        target_records = DirectMia.select_target_records(
            neighbor_threshold,
            probability_threshold,
            background_knowledge_size,
            training_set_size,
            distances,
        )
        return target_records

    @staticmethod
    def attack(
        target_records,
        attack_data,
        attack_labels_cat,
        reference_models,
        target_model,
        cut_off_p_value,
    ):
        """Attack the target records by exploiting the target model.

        1. Infer log losses of reference models.
        2. Infer log losses of target model.
        3. Sample reference losses, approximate empirical cumulative distribution
           function, smooth ecdf with piecewise cubic interpolation.
        4. Determine members and non-members with left-tailed hypothesis test.

        Parameters
        ----------
        target_records : numpy.ndarray
            Array of target records.
        attack_data : numpy.ndarray
            Array of target samples.
        attack_labels_cat : numpy.ndarray
            Array of labels (one-hot encoding) to the target samples.
        reference_models : numpy.ndarray
            Array of reference models.
        target_model : tensorflow.keras.model
            The target model.
        cut_off_p_value : float
            Cut-off-p-value used for the hypothesis test.

        Returns
        -------
        tuple
            Member and non-member of the training set of the target model and the
            p-values of the hypothesis test.
        """
        print("Infer log losses of reference models")
        reference_inferences = DirectMia.get_model_inference(
            target_records, attack_data, attack_labels_cat, reference_models
        )

        print("Infer log losses of target model")
        target_inferences = DirectMia.get_model_inference(
            target_records, attack_data, attack_labels_cat, target_model
        )

        print(
            "Sample reference losses, approximate empirical cumulative "
            + "distribution function, smooth ecdf with piecewise cubic"
            + "interpolation"
        )
        (
            used_target_records,
            pchip_references,
            ecdf_references,
        ) = DirectMia.sample_reference_losses(
            target_records, reference_inferences, True
        )

        print("Determine members and non-members with left-tailed hypothesis test")
        members, non_members, p_values = DirectMia.hypothesis_test(
            cut_off_p_value, pchip_references, target_inferences, used_target_records
        )

        return members, non_members, p_values
