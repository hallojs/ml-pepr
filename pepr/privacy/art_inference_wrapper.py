from abc import abstractmethod
import logging
import numpy as np
import os

from pepr.attack import Attack
from pepr import report
import matplotlib.pyplot as plt
from pylatex import Command, Tabular, MiniPage, NoEscape, Figure
from pylatex.section import Subsubsection

from art.estimators.classification import KerasClassifier
from art.attacks.inference import membership_inference, model_inversion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

plt.style.use("default")
# force line grid to be behind bar plots
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"


class BaseMembershipInferenceAttack(Attack):
    """
    Base ART membership inference attack class implementing the logic for running an
    membership inference attack and generating a report.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Record-indices for inference and evaluation:

        * train_record_indices (np.ndarray): (optional) Input to training process.
          Includes all features used to train the original model.
        * test_record_indices (np.ndarray): (optional) Test records that are not used in
          the training of the target model.
        * attack_record_indices (np.ndarray): Input to attack. Includes all features
          except the attacked feature.
        * attack_membership_status (np.ndarray): True membership status of the
          attack_records 1 indicates a member and 0 indicates non-member. This is used
          to compare the attacks results with the true membership status.

    target_models : iterable
        List of target models which should be tested.
    inference_attacks : list(art.attacks.Attack)
        List of ART inference attack objects per target model which are wrapped in
        this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"attack_model_type": "Attack model type"} for the attribute named
        "attack_model_type" of MembershipInferenceBlackBox.

    Attributes
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Inference attack specific attack parameters:
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
    target_models : iterable
        List of target models which should be tested.
    data_conf : dict
        Record-indices for inference and evaluation:
    inference_attacks : list(art.attacks.Attack)
        List of ART attack objects per target model which are wrapped in this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"attack_model_type": "Attack model type"} for the attribute named
        "attack_model_type" of MembershipInferenceBlackBox.
    attack_results : dict
        Dictionary storing the attack model results.

        * membership (list): List holding the inferred membership status, 1 indicates a
          member and 0 indicates non-member per target model.
        * tn (list): Number of true negatives per target model.
        * tp (list): Number of true positives per target model.
        * fn (list): Number of false negatives per target model.
        * fp (list): Number of false positives per target model.
        * precision (list): Attack precision per target model.
        * recall (list): Attack recall per target model.
        * accuracy (list): Attack accuracy per target model.
    """

    def __init__(
        self,
        attack_alias,
        data,
        labels,
        data_conf,
        target_models,
        inference_attacks,
        pars_descriptors,
    ):
        super().__init__(
            attack_alias,
            {},
            data,
            labels,
            data_conf,
            target_models,
        )

        self.inference_attacks = inference_attacks
        self.pars_descriptors = pars_descriptors

    @abstractmethod
    def inference_run(self, tm_index):
        """
        Run an attack procedure.

        Parameters
        ----------
        tm_index : int
            Index of the currently attacked target model.

        Returns
        -------
        An array holding the inferred membership status, 1 indicates a member and 0
        indicates non-member, or class probabilities.
        """
        logger.error(
            "The base class does not implement an attack! Please use an "
            "attack class like 'MembershipInferenceBlackBox'."
        )

    def run(self):
        """
        Run ART inference attack
        """
        membership = []
        acc = []
        tn_list = []
        tp_list = []
        fn_list = []
        fp_list = []
        precision_list = []
        recall_list = []

        for i, inference_attack in enumerate(self.inference_attacks):
            logger.info(f"Attack target model ({i + 1}/{len(self.inference_attacks)}).")

            membership_results = self.inference_run(i)

            true_mem = self.data_conf["attack_membership_status"][i]

            # Evaluate attack model predictions
            tn = np.count_nonzero((membership_results == 0) & (true_mem == False))
            tp = np.count_nonzero((membership_results == 1) & (true_mem == True))
            fn = np.count_nonzero((membership_results == 0) & (true_mem == True))
            fp = np.count_nonzero((membership_results == 1) & (true_mem == False))
            precision = tp / (tp + fp) if (tp + fp) else 1
            recall = tp / (fn + tp) if (fn + tp) else 0

            tn_list.append(tn)
            tp_list.append(tp)
            fn_list.append(fn)
            fp_list.append(fp)
            precision_list.append(precision)
            recall_list.append(recall)

            logger.debug(f"tn: {tn}")
            logger.debug(f"tp: {tp}")
            logger.debug(f"fn: {fn}")
            logger.debug(f"fp: {fp}")
            logger.debug(f"precision: {precision}")
            logger.debug(f"recall: {recall}")

            # Accuracy
            cmp = np.equal(true_mem, membership_results)
            accuracy = np.sum(cmp) / len(cmp)

            membership.append(membership_results)
            acc.append(accuracy)

        self.attack_results["membership"] = membership
        self.attack_results["tn"] = tn_list
        self.attack_results["tp"] = tp_list
        self.attack_results["fn"] = fn_list
        self.attack_results["fp"] = fp_list
        self.attack_results["precision"] = precision_list
        self.attack_results["recall"] = recall_list
        self.attack_results["accuracy"] = acc

        # Print every epsilon result of attack
        def _target_model_rows():
            string = ""
            for tm_i in range(len(self.target_models)):
                string = string + f"\n{f'Target Model {tm_i + 1}:':<20}"
                string = (
                    string
                    + f"{str(round(self.attack_results['accuracy'][tm_i], 3)):>10}"
                    + f"{str(round(self.attack_results['precision'][tm_i], 3)):>10}"
                    + f"{str(round(self.attack_results['recall'][tm_i], 3)):>10}"
                )
            return string

        logger.info(
            "Attack Summary"
            f"\n"
            f"\n###################### Attack Results ######################"
            f"\n" + f"\n{'Target Models':<20}"
            f"{'Accuracy':>10}"
            f"{'Precision':>10}"
            f"{'Recall':>10}" + _target_model_rows()
        )

    def create_attack_report(self, save_path="art_report", pdf=False):
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

        self.create_attack_section(save_path)
        report.report_generator(save_path, [self.report_section], pdf)

    def create_attack_section(self, save_path):
        """
        Create an attack section for the given attack instantiation.

        Parameters
        ----------
        save_path : str
            Path to save the tex, pdf and asset files of the attack report.
        """
        self._report_attack_configuration()
        self._report_attack_results()

    def _report_attack_configuration(self):
        """
        Create subsubsection about the attack and data configuration.
        """
        # Create tables for attack parameters and the data configuration.
        tm = 0  # Specify target model

        dc = self.data_conf
        self.report_section.append(Subsubsection("Attack Details"))
        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                temp_pars_desc = self.pars_descriptors.copy()
                if "attack_model" in temp_pars_desc:
                    del temp_pars_desc["attack_model"]

                values = self.inference_attacks[tm].__dict__.copy()
                if hasattr(self, "hopskipjump_args"):
                    logger.debug("Include HopSkipJump params")
                    values.update(self.hopskipjump_args)
                report.create_attack_pars_table(
                    self.report_section,
                    values,
                    temp_pars_desc,
                )

            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                # -- Create table for the data configuration
                self.report_section.append(Command("centering"))
                nr_targets, attack_data_size = dc["attack_record_indices"].shape
                with self.report_section.create(Tabular("|l|c|")) as tab_dc:
                    tab_dc.add_hline()
                    tab_dc.add_row(["Attacked target models", nr_targets])
                    tab_dc.add_hline()
                    if "train_record_indices" in dc:
                        _, train_data_size = dc["train_record_indices"].shape
                        tab_dc.add_row(["Training set size", train_data_size])
                        tab_dc.add_hline()
                    if "test_record_indices" in dc:
                        _, test_data_size = dc["test_record_indices"].shape
                        tab_dc.add_row(["Test set size", test_data_size])
                        tab_dc.add_hline()
                    tab_dc.add_row(["Attack set size", attack_data_size])
                    tab_dc.add_hline()
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "table",
                        extra_arguments="Target and Data Configuration",
                    )
                )

    def _report_attack_results(self):
        """
        Create subsubsection describing the most important results of the attack.

        This subsection contains results only for the first target model.
        """
        tm = 0  # Specify target model
        self.report_section.append(Subsubsection("Attack Results"))
        res = self.attack_results

        with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
            self.report_section.append(Command("centering"))
            with self.report_section.create(Tabular("|l|c|")) as result_tab:
                result_tab.add_hline()
                result_tab.add_row(["True Positives", round(res["tp"][tm], 3)])
                result_tab.add_hline()
                result_tab.add_row(["True Negatives", round(res["tn"][tm], 3)])
                result_tab.add_hline()
                result_tab.add_row(["False Positives", round(res["fp"][tm], 3)])
                result_tab.add_hline()
                result_tab.add_row(["False Negatives", round(res["fn"][tm], 3)])
                result_tab.add_hline()
                result_tab.add_row(["Accuracy", round(res["accuracy"][tm], 3)])
                result_tab.add_hline()
                result_tab.add_row(["Precision", round(res["precision"][tm], 3)])
                result_tab.add_hline()
                result_tab.add_row(["Recall", round(res["recall"][tm], 3)])
                result_tab.add_hline()

            self.report_section.append(Command("captionsetup", "labelformat=empty"))
            self.report_section.append(
                Command("captionof", "table", extra_arguments="Attack Summary")
            )


class MembershipInferenceBlackBox(BaseMembershipInferenceAttack):
    """
    art.attacks.inference.membership_inference.MembershipInferenceBlackBox wrapper
    class.

    Attack description:
    Implementation of a learned black-box membership inference attack.

    This implementation can use as input to the learning process probabilities/logits or
    losses, depending on the type of model and provided configuration.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * attack_model_type (str): (optional) the type of default attack model to train,
          optional. Should be one of nn (for neural network, default), rf (for random
          forest) or gb (gradient boosting). If attack_model is supplied, this option
          will be ignored.
        * input_type (str): (optional) the type of input to train the attack on. Can be
          one of: ‘prediction’ or ‘loss’. Default is prediction. Predictions can be
          either probabilities or logits, depending on the return type of the model.
        * attack_model: The attack model to train. Due to stability issues only
          TensorFlow Keras models are currently allowed. (Use PyTorch at your own risk,
          your runtime may crash.)

    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Dictionary describing for every target model which record-indices should be used
        for the attack.

        * train_record_indices (np.ndarray): Input to training process. Includes all
          features used to train the original model.
        * test_record_indices (np.ndarray): Test records that are not used in the
          training of the target model.
        * attack_record_indices (np.ndarray): Input to attack. Includes all features
          except the attacked feature.
        * attack_membership_status (np.ndarray): True membership status of the
          attack_records 1 indicates a member and 0 indicates non-member. This is used
          to compare the attacks results with the true membership status.

    target_models : iterable
        List of target models which should be tested.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        pars_descriptors = {
            "attack_model_type": "Attack model type",
            "input_type": "Input type",
            "attack_model": "Attack model",
        }

        # Display warning if no Keras model is provided
        if (
            "attack_model" in attack_pars
            and not isinstance(attack_pars["attack_model"], KerasClassifier)
        ) or "attack_model" not in attack_pars:
            logger.warning(
                "The provided Attack Model (attack_model) seems not to be a Keras "
                "classifier. This may result in stability issues and your runtime may "
                "crash! It is recommended to use a Keras model for this attack."
            )

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]

        inference_attacks = []
        for target_model in target_models:
            target_classifier = KerasClassifier(target_model, clip_values=(0, 1))
            inference_attacks.append(
                membership_inference.MembershipInferenceBlackBox(
                    classifier=target_classifier, **params
                )
            )
            if inference_attacks[-1].attack_model_type is "None":
                inference_attacks[-1].attack_model_type = "Custom"

        super().__init__(
            attack_alias,
            data,
            labels,
            data_conf,
            target_models,
            inference_attacks,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Membership Inference Black-Box",
            self.attack_alias,
            "ART_MembershipInferenceBlackBox",
        )

    def __getstate__(self):
        if self.inference_attacks[0].default_model:
            for inference_attack in self.inference_attacks:
                # Delete attack model, because some are not pickable
                del inference_attack.__dict__["attack_model"]
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        # TODO: Restore attack models (not required for generating report)
        # The attack_model attribute of the object must be stored and restored
        # separately due to the incompatibility to Python's pickle. This can be done
        # with model.save() (TensorFlow) or torch.save() (PyTorch). Restoring the
        # attack model is not required for generating a report like the attack runner
        # does.

    def inference_run(self, tm_index):
        """
        Run the black box inference attack.

        Parameters
        ----------
        tm_index : int
            Index of the currently attacked target model.

        Returns
        -------
        An array holding the inferred membership status, 1 indicates a member and 0
        indicates non-member, or class probabilities.
        """
        # Train attack model
        inference_attack = self.inference_attacks[tm_index]
        inference_attack.fit(
            self.data[self.data_conf["train_record_indices"][tm_index]],
            self.labels[self.data_conf["train_record_indices"][tm_index]],
            self.data[self.data_conf["test_record_indices"][tm_index]],
            self.labels[self.data_conf["test_record_indices"][tm_index]],
        )

        # Get inference
        membership_results = inference_attack.infer(
            self.data[self.data_conf["attack_record_indices"][tm_index]],
            self.labels[self.data_conf["attack_record_indices"][tm_index]],
        )

        return membership_results


class MembershipInferenceBlackBoxRuleBased(BaseMembershipInferenceAttack):
    """
    art.attacks.inference.membership_inference.MembershipInferenceBlackBoxRuleBased wrapper
    class.

    Attack description:
    Implementation of a simple, rule-based black-box membership inference attack.

    This implementation uses the simple rule: if the model’s prediction for a sample is
    correct, then it is a member. Otherwise, it is not a member.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters: (no parameters)
    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Dictionary describing for every target model which record-indices should be used
        for the attack.

        * attack_record_indices (np.ndarray): Input to attack. Includes all features
          except the attacked feature.
        * attack_membership_status (np.ndarray): True membership status of the
          attack_records 1 indicates a member and 0 indicates non-member. This is used
          to compare the attacks results with the true membership status.

    target_models : iterable
        List of target models which should be tested.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        pars_descriptors = {}  # No additional parameters

        inference_attacks = []
        for target_model in target_models:
            target_classifier = KerasClassifier(target_model, clip_values=(0, 1))
            inference_attacks.append(
                membership_inference.MembershipInferenceBlackBoxRuleBased(
                    classifier=target_classifier
                )
            )

        super().__init__(
            attack_alias,
            data,
            labels,
            data_conf,
            target_models,
            inference_attacks,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Membership Inference Black-Box Rule-Based",
            self.attack_alias,
            "ART_MembershipInferenceBlackBoxRuleBased",
        )

    def inference_run(self, tm_index):
        """
        Run the rule based black box inference attack.

        Parameters
        ----------
        tm_index : int
            Index of the currently attacked target model.

        Returns
        -------
        An array holding the inferred membership status, 1 indicates a member and 0
        indicates non-member, or class probabilities.
        """
        inference_attack = self.inference_attacks[tm_index]
        membership_results = inference_attack.infer(
            self.data[self.data_conf["attack_record_indices"][tm_index]],
            self.labels[self.data_conf["attack_record_indices"][tm_index]],
        )

        return membership_results


class LabelOnlyDecisionBoundary(BaseMembershipInferenceAttack):
    """
    art.attacks.inference.membership_inference.LabelOnlyDecisionBoundary wrapper
    class.

    Attack description:
    Implementation of Label-Only Inference Attack based on Decision Boundary.

    Paper link: https://arxiv.org/abs/2007.14321

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * distance_threshold_tau (float): Threshold distance for decision boundary.
          Samples with boundary distances larger than threshold are considered members
          of the training dataset.
        * norm: (optional) Order of the norm. Possible values: “inf”, np.inf or 2.
        * max_iter (int): (optional) Maximum number of iterations.
        * max_eval (int): (optional) Maximum number of evaluations for estimating
          gradient.
        * init_eval (int): (optional) Initial number of evaluations for estimating
          gradient.
        * init_size (int): (optional) Maximum number of trials for initial generation of
          adversarial examples.
        * verbose (bool): (optional) Show progress bars.

    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Dictionary describing for every target model which record-indices should be used
        for the attack.

        * train_record_indices (np.ndarray): Input to training process. Includes all
          features used to train the original model.
        * test_record_indices (np.ndarray): Test records that are not used in the
          training of the target model.
        * attack_record_indices (np.ndarray): Input to attack. Includes all features
          except the attacked feature.
        * attack_membership_status (np.ndarray): True membership status of the
          attack_records 1 indicates a member and 0 indicates non-member. This is used
          to compare the attacks results with the true membership status.

    target_models : iterable
        List of target models which should be tested.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        pars_descriptors = {
            "distance_threshold_tau": "Threshold distance",
            # HopSkipJump parameters
            "norm": "Adversarial perturbation norm",
            "max_iter": "Max. iterations",
            "max_eval": "Max. evaluations",
            "init_eval": "Initial evaluations",
            "init_size": "Max. trials",
            "verbose": "Verbose output",
        }

        # Save HopSkipJump parameters
        self.hopskipjump_args = attack_pars.copy()
        del self.hopskipjump_args["distance_threshold_tau"]
        # Hide verbose parameter from report
        if "verbose" in self.hopskipjump_args:
            del self.hopskipjump_args["verbose"]

        inference_attacks = []
        for target_model in target_models:
            target_classifier = KerasClassifier(target_model, clip_values=(0, 1))
            inference_attacks.append(
                membership_inference.LabelOnlyDecisionBoundary(
                    target_classifier,
                    distance_threshold_tau=attack_pars["distance_threshold_tau"],
                )
            )

        super().__init__(
            attack_alias,
            data,
            labels,
            data_conf,
            target_models,
            inference_attacks,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Label Only Decision Boundary",
            self.attack_alias,
            "ART_LabelOnlyDecisionBoundary",
        )

    def inference_run(self, tm_index):
        """
        Run the Label-Only - Decision Boundary inference attack.

        Parameters
        ----------
        tm_index : int
            Index of the currently attacked target model.

        Returns
        -------
        An array holding the inferred membership status, 1 indicates a member and 0
        indicates non-member, or class probabilities.
        """
        inference_attack = self.inference_attacks[tm_index]
        # Calibrate distance threshold maximising the membership inference accuracy on
        # x_train and x_test.
        inference_attack.calibrate_distance_threshold(
            self.data[self.data_conf["train_record_indices"][tm_index]],
            self.labels[self.data_conf["train_record_indices"][tm_index]],
            self.data[self.data_conf["test_record_indices"][tm_index]],
            self.labels[self.data_conf["test_record_indices"][tm_index]],
            **self.hopskipjump_args,
        )

        # Get inference
        membership_results = inference_attack.infer(
            self.data[self.data_conf["attack_record_indices"][tm_index]],
            self.labels[self.data_conf["attack_record_indices"][tm_index]],
            **self.hopskipjump_args,
        )

        return membership_results


class MIFace(Attack):
    """
    art.attacks.inference.membership_inference.MIFace wrapper class.

    Attack description:
    Implementation of the MIFace algorithm from Fredrikson et al. (2015). While in that
    paper the attack is demonstrated specifically against face recognition models, it is
    applicable more broadly to classifiers with continuous features which expose class
    gradients.

    Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * max_iter (int): (optional) Maximum number of gradient descent iterations for
          the model inversion.
        * window_length (int): (optional) Length of window for checking whether descent
          should be aborted.
        * threshold (float): (optional) Threshold for descent stopping criterion.
        * batch_size (int): (optional) Size of internal batches.
        * verbose (bool): (optional) Show progress bars.

    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Dictionary describing for every target model which record-indices should be used
        for the attack.

        * initial_input_data (np.ndarray): An array with the initial input to the victim
          classifier. If None, then initial input will be initialized as zero array.
        * initial_input_targets (np.ndarray): Target values of shape (nb_samples,).

    target_models : iterable
        List of target models which should be tested.

    Attributes
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Inference attack specific attack parameters:
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
    target_models : iterable
        List of target models which should be tested.
    data_conf : dict
        Record-indices for inference and evaluation:
    inference_attacks : list(art.attacks.Attack)
        List of ART attack objects per target model which are wrapped in this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"max_iter": "Max. iterations"} for the attribute named
        "max_iter" of MIFace.
    attack_results : dict
        Dictionary storing the attack model results.

        * inferred_training_samples (list): The inferred training samples per target
          model.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        super().__init__(
            attack_alias,
            {},
            data,
            labels,
            data_conf,
            target_models,
        )

        self.pars_descriptors = {
            "max_iter": "Max. iterations",
            "window_length": "Window length",
            "threshold": "Stopping threshold",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in self.pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]

        self.inference_attacks = []
        for target_model in target_models:
            target_classifier = KerasClassifier(target_model, clip_values=(0, 1))
            self.inference_attacks.append(
                model_inversion.MIFace(target_classifier, **params)
            )

        self.report_section = report.ReportSection(
            "Model Inversion MIFace",
            self.attack_alias,
            "ART_MIFace",
        )

    def run(self):
        """
        Run ART inference attack
        """
        inferred_samples = []

        for i, inference_attack in enumerate(self.inference_attacks):
            logger.info(f"Attack target model ({i + 1}/{len(self.inference_attacks)}).")

            inferred_res = inference_attack.infer(
                self.data_conf["initial_input_data"][i],
                self.data_conf["initial_input_targets"][i],
            )

            inferred_samples.append(inferred_res)

        self.attack_results["inferred_training_samples"] = inferred_samples

        nb_tm = len(self.target_models)
        if nb_tm > 1:
            desc = "Fist inferred training sample per target model:"
        else:
            desc = "Fist inferred training samples:"
        logger.info(
            "Attack Summary"
            f"\n"
            f"\n###################### Attack Results ######################"
            f"\n{desc}"
        )

        plt.figure(figsize=(15, 5))
        if nb_tm > 1:
            for i in range(nb_tm):
                plt.subplot(1, nb_tm, i + 1)
                plt.axis("off")
                plt.imshow(inferred_samples[i][0])
        else:
            image_count = min(10, len(inferred_samples[0]))
            for i in range(image_count):
                plt.subplot(1, image_count, i + 1)
                plt.axis("off")
                plt.imshow(inferred_samples[0][i])

    def create_attack_report(self, save_path="art_report", pdf=False):
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

        self.create_attack_section(save_path)
        report.report_generator(save_path, [self.report_section], pdf)

    def create_attack_section(self, save_path):
        """
        Create an attack section for the given attack instantiation.

        Parameters
        ----------
        save_path : str
            Path to save the tex, pdf and asset files of the attack report.
        """
        self._report_attack_configuration()
        self._report_attack_results(save_path)

    def _report_attack_configuration(self):
        """
        Create subsubsection about the attack and data configuration.
        """
        # Create tables for attack parameters and the data configuration.
        tm = 0  # Specify target model

        dc = self.data_conf
        self.report_section.append(Subsubsection("Attack Details"))
        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                temp_pars_desc = self.pars_descriptors.copy()
                if "verbose" in temp_pars_desc:
                    del temp_pars_desc["verbose"]
                values = self.inference_attacks[tm].__dict__.copy()

                report.create_attack_pars_table(
                    self.report_section,
                    values,
                    temp_pars_desc,
                )

            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                # -- Create table for the data configuration
                self.report_section.append(Command("centering"))
                with self.report_section.create(Tabular("|l|c|")) as tab_dc:
                    tab_dc.add_hline()
                    tab_dc.add_row(["Attacked target models", len(self.target_models)])
                    tab_dc.add_hline()
                    if "initial_input_data" in dc:
                        input_size = dc["initial_input_data"][tm].shape[0]
                        tab_dc.add_row(["Input data size", input_size])
                        tab_dc.add_hline()
                    if "initial_input_targets" in dc:
                        target_size = len(dc["initial_input_targets"][tm])
                        tab_dc.add_row(["Target size", target_size])
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

        # Plot image grid
        nb_smaples = min(10, len(res["inferred_training_samples"][tm]))
        ncols = min(10, nb_smaples)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(15, 5))
        for i in range(nb_smaples):
            image = res["inferred_training_samples"][tm][i]
            ax = axes[i]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"Example {i}")
            ax.imshow(image)

        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        fig.savefig(
            save_path + f"/fig/{alias_no_spaces}-examples.pdf", bbox_inches="tight"
        )
        plt.close(fig)

        with self.report_section.create(Figure(position="H")) as fig:
            fig.add_image(
                f"fig/{alias_no_spaces}-examples.pdf", width=NoEscape(r"\textwidth")
            )
            self.report_section.append(Command("captionsetup", "labelformat=empty"))
            self.report_section.append(
                Command(
                    "captionof",
                    "figure",
                    extra_arguments="Inferred Samples (up to 10 samples, not sorted, "
                    "not selected by any criteria)",
                )
            )
