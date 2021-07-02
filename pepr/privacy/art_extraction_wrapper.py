import logging
import numpy as np
import os

from pepr.attack import Attack
from pepr import report
import matplotlib.pyplot as plt
from pylatex import Command, Tabular, MiniPage, NoEscape
from pylatex.section import Subsubsection

import art
from art.estimators.classification import KerasClassifier

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

plt.style.use("default")
# force line grid to be behind bar plots
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"


class BaseExtractionAttack(Attack):
    """
    Base ART extraction attack class implementing the logic for running an extraction
    attack and generating a report.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Extraction attack specific attack parameters:

        * stolen_models (list): List of untrained input models for every target model to
          store the stolen training data in.

    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Record-indices for extraction and evaluation:

        * stolen_record_indices (np.ndarray): Indices of records to use for the
          extraction attack.
        * eval_record_indices (np.ndarray): Indices of records for measuring the
          accuracy of the extracted model.

    target_models : iterable
        List of target models which should be tested.
    extraction_attacks : list(art.attacks.Attack)
        List of ART extraction attack objects per target model which are wrapped in
        this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"classifier": "A victim classifier"} for the attribute named
        "classifier" of CopycatCNN.

    Attributes
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Extraction attack specific attack parameters:
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
    target_models : iterable
        List of target models which should be tested.
    data_conf : dict
        Record-indices for extraction and evaluation:
    extraction_attacks : list(art.attacks.Attack)
        List of ART attack objects per target model which are wrapped in this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"classifier": "A victim classifier"} for the attribute named
        "classifier" of CopycatCNN.
    attack_results : dict
        Dictionary storing the attack model results.

        * extracted_classifiers (list): List of extracted classifiers per target model.
        * ec_accuracy (list): List of the accuracy of the extracted classifiers per
          target model.
        * ec_accuracy_list (list): List of the accuracy of the extracted classifiers per
          target model and class. Shape: (target_model, class)
    """

    def __init__(
        self,
        attack_alias,
        attack_pars,
        data,
        labels,
        data_conf,
        target_models,
        extraction_attacks,
        pars_descriptors,
    ):
        super().__init__(
            attack_alias,
            attack_pars,
            data,
            labels,
            data_conf,
            target_models,
        )

        self.extraction_attacks = extraction_attacks
        self.pars_descriptors = pars_descriptors
        self.classifiers = [x.estimator for x in extraction_attacks]

    def run(self):
        """
        Run ART model extraction
        """
        stolen_classifiers = [
            KerasClassifier(sm, clip_values=(0, 1))
            for sm in self.attack_pars["stolen_models"]
        ]
        extracted_classifiers = []
        ec_accuracy = []
        ec_accuracy_list = []
        for i, extraction_attack in enumerate(self.extraction_attacks):
            logger.info(
                f"Attack target model ({i + 1}/{len(self.extraction_attacks)})."
            )
            stolen_data = self.data[self.data_conf["stolen_record_indices"][i]]
            stolen_labels = self.labels[self.data_conf["stolen_record_indices"][i]]

            extracted_classifier = extraction_attack.extract(
                stolen_data,
                stolen_labels,
                thieved_classifier=stolen_classifiers[i],
            )

            # Get accuracy of extracted model for every class
            eval_data = self.data[self.data_conf["eval_record_indices"][i]]
            eval_labels = self.labels[self.data_conf["eval_record_indices"][i]]
            ec_acc_tm = []
            for j in range(np.max(eval_labels) + 1):
                idx, = np.where(eval_labels == j)
                if idx.size == 0:
                    ec_acc_tm.append(np.NaN)
                else:
                    _, acc = extracted_classifier._model.evaluate(
                        eval_data[idx], eval_labels[idx]
                    )
                    ec_acc_tm.append(acc)

            extracted_classifiers.append(extracted_classifier)
            ec_accuracy.append(np.nanmean(ec_acc_tm))
            ec_accuracy_list.append(ec_acc_tm)

        self.attack_results["extracted_classifiers"] = extracted_classifiers
        self.attack_results["ec_accuracy"] = ec_accuracy
        self.attack_results["ec_accuracy_list"] = ec_accuracy_list

        # Print every epsilon result of attack
        def _target_model_rows():
            string = ""
            for tm_i in range(len(self.target_models)):
                string = string + f"\n{f'Target Model {tm_i + 1}:':<20}"
                string = (
                    string
                    + f"{str(round(self.attack_results['ec_accuracy'][tm_i], 3)):>10}"
                )
            return string

        logger.info(
            "Attack Summary"
            f"\n"
            f"\n###################### Attack Results ######################"
            f"\n"
            + f"\n{'Target Models':<20}{'Accuracy':>10}"
            + _target_model_rows()
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
        self._report_attack_results(save_path)

    def _gen_attack_pars_rows(self, tm, table):
        """
        Generate LaTex table rows with fancy parameter descriptions.

        Parameters
        ----------
        tm : int
            Target model index.
        table : pylatex.Tabular
            Pylatex table in which the rows are append.
        """
        for key in self.pars_descriptors:
            desc = self.pars_descriptors[key]
            if key == "verbose":
                continue
            value = str(self.extraction_attacks[tm].__dict__[key])

            table.add_hline()
            table.add_row([desc, value])

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
                # -- Create table for the attack parameters.
                self.report_section.append(Command("centering"))
                temp_pars_desc = self.pars_descriptors.copy()
                if "verbose" in temp_pars_desc:
                    del temp_pars_desc["verbose"]
                values = self.extraction_attacks[tm].__dict__.copy()
                report.create_attack_pars_table(
                    self.report_section,
                    values,
                    temp_pars_desc,
                )

            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                # -- Create table for the data configuration
                self.report_section.append(Command("centering"))
                nr_targets, attack_data_size = dc["stolen_record_indices"].shape
                with self.report_section.create(Tabular("|l|c|")) as tab_dc:
                    tab_dc.add_hline()
                    tab_dc.add_row(["Attacked target models", nr_targets])
                    tab_dc.add_hline()
                    tab_dc.add_row(["Attack set size", attack_data_size])
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        ["Evaluation set size", len(dc["eval_record_indices"][tm])]
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

        # Histogram
        path = report.plot_class_dist_histogram(
            self.attack_alias, res["ec_accuracy_list"][tm], save_path
        )

        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                self.report_section.append(
                    Command(
                        "includegraphics",
                        NoEscape(path),
                        "width=8cm",
                    )
                )
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "figure",
                        extra_arguments="Evaluation Accuracy Distribution",
                    )
                )

            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                with self.report_section.create(Tabular("|l|c|")) as result_tab:
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["Extracted model accuracy", round(res["ec_accuracy"][tm], 3)]
                    )
                    result_tab.add_hline()

                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command("captionof", "table", extra_arguments="Attack Summary")
                )


class CopycatCNN(BaseExtractionAttack):
    """
    art.attacks.extraction.CopycatCNN wrapper class.

    Attack description:
    Implementation of the Copycat CNN attack from Rodrigues Correia-Silva et al. (2018).

    Paper link: https://arxiv.org/abs/1806.05476

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * batch_size_fit (int): (optional) Size of batches for fitting the thieved
          classifier.
        * batch_size_query (int): (optional) Size of batches for querying the victim
          classifier.
        * nb_epochs (int): (optional) Number of epochs to use for training.
        * nb_stolen (int): (optional) Number of queries submitted to the victim
          classifier to steal it.
        * use_probability (bool): (optional) Use probability.
        * stolen_record_indices (np.ndarray): Indices of records to use for the
          extraction attack.

    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Dictionary describing for every target model which record-indices should be used
        for the attack.

        * stolen_record_indices (np.ndarray): Indices of records to use for the
          extraction attack.
        * eval_record_indices (np.ndarray): Indices of records for measuring the
          accuracy of the extracted model.

    target_models : iterable
        List of target models which should be tested.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        pars_descriptors = {
            "batch_size_fit": "Batch size (thieved classifier)",
            "batch_size_query": "Batch size (victim classifier)",
            "nb_epochs": "Number of epochs for training",
            "nb_stolen": "Number of victim queries",
            "use_probability": "Use probability",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]

        extraction_attacks = []
        for target_model in target_models:
            target_classifier = KerasClassifier(target_model, clip_values=(0, 1))
            extraction_attacks.append(
                art.attacks.extraction.CopycatCNN(
                    classifier=target_classifier, **params
                )
            )

        super().__init__(
            attack_alias,
            {"stolen_models": attack_pars["stolen_models"]},
            data,
            labels,
            data_conf,
            target_models,
            extraction_attacks,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Copycat CNN",
            self.attack_alias,
            "ART_CopycatCNN",
        )


class KnockoffNets(BaseExtractionAttack):
    """
    art.attacks.extraction.KnockoffNets wrapper class.

    Attack description:
    Implementation of the Knockoff Nets attack from Orekondy et al. (2018).

    Paper link: https://arxiv.org/abs/1812.02766

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * batch_size_fit (int): (optional) Size of batches for fitting the thieved
          classifier.
        * batch_size_query (int): (optional) Size of batches for querying the victim
          classifier.
        * nb_epochs (int): (optional) Number of epochs to use for training.
        * nb_stolen (int): (optional) Number of queries submitted to the victim
          classifier to steal it.
        * use_probability (bool): (optional) Use probability.
        * sampling_strategy (str): Sampling strategy, either `random` or `adaptive`.
        * reward (str): Reward type, in ['cert', 'div', 'loss', 'all'].
        * verbose (bool): Show progress bars.
        * stolen_record_indices (np.ndarray): Indices of records to use for the
          extraction attack.

    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Dictionary describing for every target model which record-indices should be used
        for the attack.

        * stolen_record_indices (np.ndarray): Indices of records to use for the
          extraction attack.
        * eval_record_indices (np.ndarray): Indices of records for measuring the
          accuracy of the extracted model.

    target_models : iterable
        List of target models which should be tested.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        pars_descriptors = {
            "batch_size_fit": "Batch size (thieved classifier)",
            "batch_size_query": "Batch size (victim classifier)",
            "nb_epochs": "Number of epochs for training",
            "nb_stolen": "Number of victim queries",
            "use_probability": "Use probability",
            "sampling_strategy": "Sampling strategy",
            "reward": "Reward type",
            "verbose": "Show progress bars",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]

        extraction_attacks = []
        for target_model in target_models:
            target_classifier = KerasClassifier(target_model, clip_values=(0, 1))
            extraction_attacks.append(
                art.attacks.extraction.KnockoffNets(
                    classifier=target_classifier, **params
                )
            )

        super().__init__(
            attack_alias,
            {"stolen_models": attack_pars["stolen_models"]},
            data,
            labels,
            data_conf,
            target_models,
            extraction_attacks,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Knockoff Nets",
            self.attack_alias,
            "ART_KnockoffNets",
        )
