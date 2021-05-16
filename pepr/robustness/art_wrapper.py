"""PePR wrapper classes for ART attack classes."""

import logging
import numpy as np
import os

from pepr.attack import Attack
from pepr import report
import matplotlib.pyplot as plt
from pylatex import Command, Tabular, MiniPage
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


class BaseEvasionAttack(Attack):
    """
    Base ART attack class implementing the logic for running an evasion attack and
    generating a report.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    use_labels : bool
        If true, the true labels are used as targets.
    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    attack_indices_per_target : numpy.ndarray
        Array of indices to attack per target model.
    target_models : iterable
        List of target models which should be tested.
    art_attacks : list(art.attacks.Attack)
        List of ART attack objects per target model which are wrapped in this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"norm": "Adversarial perturbation norm"} for the attribute named
        "norm" of FastGradientMethod.

    Attributes
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    use_labels : bool
        If true, the true labels are used as targets.
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
    target_models : iterable
        List of target models which should be tested.
    attack_indices_per_target : numpy.ndarray
        Array of indices to attack per target model.
    art_attacks : list(art.attacks.Attack)
        List of ART attack objects per target model which are wrapped in this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"norm": "Adversarial perturbation norm"} for the attribute named
        "norm" of FastGradientMethod.
    attack_results : dict
        Dictionary storing the attack model results.

        * adversarial_examples (list): Array of adversarial examples per target model.
        * success_rate (list): Percentage of misclassified adversarial examples
          per target model.
        * l2_distance (list): Euclidean distance (L2 norm) between original and
          perturbed images per target model.
    """

    def __init__(
        self,
        attack_alias,
        use_labels,
        data,
        labels,
        attack_indices_per_target,
        target_models,
        art_attacks,
        pars_descriptors,
    ):
        super().__init__(
            attack_alias,
            {"use_labels": use_labels},
            data,
            labels,
            {"attack_indices_per_target": attack_indices_per_target},
            target_models,
        )

        self.attack_indices_per_target = attack_indices_per_target
        self.art_attacks = art_attacks
        self.pars_descriptors = pars_descriptors
        self.classifiers = [x.estimator for x in art_attacks]
        self.use_labels = use_labels

    def art_run(self, attack_index, data, labels=None):
        """
        ART attack run function.

        Parameters
        ----------
        attack_index : int
            Index of the corresponding target model.
        data : numpy.ndarray
            Dataset slice containing images to attack the corresponding target model.
        labels : numpy.ndarray
            Dataset slice with true labels to attack the corresponding target model.
        """
        if labels is not None:
            # Convert to one-hot representation
            n_classes = self.classifiers[attack_index].nb_classes
            labels = np.eye(n_classes, dtype=np.int_)[labels]
            return self.art_attacks[attack_index].generate(data, labels)

        return self.art_attacks[attack_index].generate(data)

    def run(self):
        """Run the ART attack."""
        adv_list = []
        misclass_list = []
        l2_dist_list = []

        # Run attack for every target model
        for i, art_attack in enumerate(self.art_attacks):
            logger.info(f"Attack target model ({i + 1}/{len(self.art_attacks)}).")
            data = self.data[self.attack_indices_per_target[i]]
            labels = self.labels[self.attack_indices_per_target[i]]

            if self.use_labels:
                adv = self.art_run(i, data, labels)
            else:
                adv = self.art_run(i, data)
            adv_list.append(adv)

            # Calculate accuracy on adversarial examples
            _, accuracy = self.target_models[i].evaluate(adv, labels)

            # Calculate L2 distance of adversarial examples
            l2_dist = np.linalg.norm(adv - data)

            misclass_list.append(1 - accuracy)
            l2_dist_list.append(l2_dist)

        self.attack_results["adversarial_examples"] = adv_list
        self.attack_results["success_rate"] = misclass_list
        self.attack_results["l2_distance"] = l2_dist_list

        # Print every epsilon result of attack
        def _target_model_rows():
            string = ""
            for tm_i in range(len(self.target_models)):
                string = string + f"\n{f'Target Model {tm_i + 1}:':<20}"
                string = (
                    string
                    + f"{str(round(self.attack_results['success_rate'][tm_i], 3)):>10}"
                    + f"{str(round(self.attack_results['l2_distance'][tm_i], 3)):>10}"
                )
            return string

        logger.info(
            "Attack Summary"
            f"\n"
            f"\n###################### Attack Results ######################"
            f"\n"
            + f"\n{'Target Models':<20}{'Success':>10}{'Distance':>10}"
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

    def _report_attack_configuration(self):
        """Create subsubsection about the attack and data configuration."""
        # Create tables for attack parameters and the data configuration.
        tm = 0  # Specify target model

        def gen_attack_pars_rows(table):
            for key in self.pars_descriptors:
                desc = self.pars_descriptors[key]
                if key == "targeted":
                    key = "_targeted"
                value = str(self.art_attacks[tm].__dict__[key])

                table.add_hline()
                table.add_row([desc, value])

        dc = self.data_conf
        self.report_section.append(Subsubsection("Attack Details"))
        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                # -- Create table for the attack parameters.
                self.report_section.append(Command("centering"))
                with self.report_section.create(Tabular("|l|c|")) as tab_ap:
                    tab_ap.add_hline()
                    tab_ap.add_row(["Use true labels", self.use_labels])
                    gen_attack_pars_rows(tab_ap)
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
                nr_targets, target_attack_set_size = dc[
                    "attack_indices_per_target"
                ].shape
                with self.report_section.create(Tabular("|l|c|")) as tab_dc:
                    tab_dc.add_hline()
                    tab_dc.add_row(["Attacked target models", nr_targets])
                    tab_dc.add_hline()
                    tab_dc.add_row(
                        ["Target model's attack sets size", target_attack_set_size]
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

        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                pass  # TODO: Graph or other visualization

            # Result table
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))

                with self.report_section.create(Tabular("|l|c|")) as result_tab:
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["Success Rate", round(res["success_rate"][tm], 3)]
                    )
                    result_tab.add_hline()
                    result_tab.add_row(
                        ["L2 Distance", round(res["l2_distance"][tm], 3)]
                    )
                    result_tab.add_hline()

                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command("captionof", "table", extra_arguments="Attack Summary")
                )


class FastGradientMethod(BaseEvasionAttack):
    """
    art.attacks.evasion.FastGradientMethod wrapper class.

    Attack description:
    This attack was originally implemented by Goodfellow et al. (2015) with the infinity
    norm (and is known as the “Fast Gradient Sign Method”). This implementation extends
    the attack to other norms, and is therefore called the Fast Gradient Method.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * norm: (optional) The norm of the adversarial perturbation. Possible values:
          “inf”, np.inf, 1 or 2.
        * eps (float): (optional) Attack step size (input variation).
        * eps_step (float): (optional) Step size of input variation for minimal
          perturbation computation.
        * targeted (bool): (optional) Indicates whether the attack is targeted (True) or
          untargeted (False).
        * num_random_init (int): (optional) Number of random initialisations within the
          epsilon ball. For random_init=0 starting at the original input.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * minimal (bool): (optional) Indicates if computing the minimal perturbation
          (True). If True, also define eps_step for the step size and eps for the
          maximum perturbation.
        * use_labels (bool): (optional) If true, the true labels are used as targets.

    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    data_conf : dict
        Dictionary describing for every target model which record-indices should be used
        for the attack.

        * attack_indices_per_target (numpy.ndarray): Array of indices of images to
          attack per target model.

    target_models : iterable
        List of target models which should be tested.
    """

    def __init__(
        self, attack_alias, attack_pars, data, labels, data_conf, target_models
    ):
        # Handle specific attack class parameters
        params = {}
        if "norm" in attack_pars:
            params["norm"] = attack_pars["norm"]
        if "eps" in attack_pars:
            params["eps"] = attack_pars["eps"]
        if "eps_step" in attack_pars:
            params["eps_step"] = attack_pars["eps_step"]
        if "targeted" in attack_pars:
            params["targeted"] = attack_pars["targeted"]
        if "num_random_init" in attack_pars:
            params["num_random_init"] = attack_pars["num_random_init"]
        if "batch_size" in attack_pars:
            params["batch_size"] = attack_pars["batch_size"]
        if "minimal" in attack_pars:
            params["minimal"] = attack_pars["minimal"]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(
                art.attacks.evasion.FastGradientMethod(estimator=est, **params)
            )

        pars_descriptors = {
            "norm": "Adversarial perturbation norm",
            "eps": "Step size",
            "eps_step": "Step size min. perturbation",
            "targeted": "Targeted attack",
            "num_random_init": "Random initialisations",
            "batch_size": "Batch size",
            "minimal": "Computing minimal perturbation",
        }

        super().__init__(
            attack_alias,
            use_labels,
            data,
            labels,
            data_conf["attack_indices_per_target"],
            target_models,
            art_attacks,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Fast Gradient Method",
            self.attack_alias,
            "FastGradientMethod",
        )
