"""PePR wrapper classes for foolbox attack classes."""

import logging
import numpy
import numpy as np
import os

from pepr.attack import Attack
from pepr import report
import matplotlib.pyplot as plt
from pylatex import Command, NoEscape, Tabular, MiniPage
from pylatex.section import Subsubsection
from pylatex.utils import bold

import foolbox as fb
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

plt.style.use("default")
# force line grid to be behind bar plots
plt.rcParams["axes.axisbelow"] = True
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = ":"


class BaseAttack(Attack):
    """
    Base foolbox attack class implementing the logic for running the attack and
    generating a report.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    epsilons : iterable
        List of one or more Epsilons for the attack.
    data : numpy.ndarray
        Dataset with all input images used to attack the target models.
    labels : numpy.ndarray
        Array of all labels used to attack the target models.
    attack_indices_per_target : numpy.ndarray
        Array of indices to attack per target model.
    target_models : iterable
        List of target models which should be tested.
    foolbox_attack : foolbox.attack.Attack
        The foolbox attack object which is wrapped in this class.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.
        Example: {"target": "Contrast reduction target"} for the attribute named
        "target" of L2ContrastReductionAttack.

    Attributes
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed parameters fo the attack.
    data : numpy.ndarray
        Dataset with all training samples used in the given pentesting setting.
    labels : numpy.ndarray
        Array of all labels used in the given pentesting setting.
    target_models : iterable
        List of target models which should be tested.
    fmodels : iterable
        List of foolbox models converted from target models.
    foolbox_attack : foolbox.attack.Attack
        The foolbox attack object which is wrapped in this class.
    attack_results : dict
        Dictionary storing the attack model results.

        * raw (list): List of raw adversarial examples per target model.
        * clipped (list):  The clipped adversarial examples. These are guaranteed to not
          be perturbed more than epsilon and thus are the actual adversarial examples
          you want to visualize. Note that some of them might not actually switch the
          class.
        * is_adv (list): Contains a boolean for each sample, indicating
          which samples are true adversarial that are both misclassified and within the
          epsilon balls around the clean samples. For every target model a
          tensorflow.Tensor with an array of shape (epsilons, data).
        * success_rate (list): Percentage of misclassified adversarial examples
          per target model and epsilon.
        * avg_l2_distance (list): Average euclidean distance (L2 norm) between original
          and perturbed images (epsilon is upper bound) per target model.
    """

    def __init__(
        self,
        attack_alias,
        epsilons,
        data,
        labels,
        attack_indices_per_target,
        target_models,
        foolbox_attack,
        pars_descriptors,
    ):
        super().__init__(
            attack_alias,
            {"epsilons": epsilons},
            data,
            labels,
            {"attack_indices_per_target": attack_indices_per_target},
            target_models,
        )

        self.epsilons = epsilons
        self.attack_indices_per_target = attack_indices_per_target
        self.foolbox_attack = foolbox_attack
        self.fmodels = [fb.TensorFlowModel(x, bounds=(0, 1)) for x in target_models]
        self.pars_descriptors = pars_descriptors

    def __getstate__(self):
        del self.__dict__["fmodels"]
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.fmodels = [
            fb.TensorFlowModel(x, bounds=(0, 1)) for x in self.target_models
        ]

    def run(self):
        """Run Foolbox attack."""
        # Make sure, epsilons is type list for consistency
        try:
            list(self.epsilons)
        except TypeError:
            self.epsilons = [self.epsilons]

        raw_list = []
        clipped_list = []
        is_adv_list = []
        misclass_list = []
        l2_dist_list = []

        # Run attack for every target model
        for i, fmodel in enumerate(self.fmodels):
            logger.info(f"Attack target model ({i + 1}/{len(self.fmodels)}).")
            indices = self.attack_indices_per_target[i]
            inputs_t = tf.convert_to_tensor(self.data[indices])
            criterion_t = tf.convert_to_tensor(self.labels[indices])

            raw, clipped, is_adv = self.foolbox_attack(
                fmodel, inputs_t, criterion_t, epsilons=self.epsilons
            )

            raw_list.append(raw)
            clipped_list.append(clipped)
            is_adv_list.append(is_adv)

            misclass = is_adv.numpy().mean(axis=-1)
            misclass_list.append(misclass)
            logger.debug(f"Average attack accuracy: {misclass}")

            # Calculate average distance of adversarial examples
            dist_eps = []
            for i in range(len(self.epsilons)):
                dist_eps.append(fb.distances.l2(inputs_t, raw[i]).numpy().mean())
            l2_dist_list.append(dist_eps)

        self.attack_results["raw"] = raw_list
        self.attack_results["clipped"] = clipped_list
        self.attack_results["is_adv"] = is_adv_list
        self.attack_results["success_rate"] = misclass_list
        self.attack_results["avg_l2_distance"] = l2_dist_list

        # Print attack summary
        def _list_to_formatted_string(arr):
            string = ""
            # Short output if needed
            if len(arr) <= 10:
                for item in arr:
                    string = string + f"{round(item, 3):>10}"
            else:
                for i in range(5):
                    string = string + f"{round(arr[i], 3):>10}"
                string = string + f"{'...':>10}"
                for i in range(-5, 0):
                    string = string + f"{round(arr[i], 3):>10}"
            return string

        # Print every epsilon result of attack
        def _target_model_rows():
            string = ""
            for tm_i in range(len(self.target_models)):
                string = string + f"\n{f'Target Model {tm_i + 1}:':<20}"
                string = string + _list_to_formatted_string(
                    self.attack_results["success_rate"][tm_i]
                )
            return string

        logger.info(
            "Attack Summary"
            f"\n"
            f"\n###################### Attack Results ######################"
            f"\n"
            + f"\n{'Epsilons:':<20}"
            + _list_to_formatted_string(self.epsilons)
            + _target_model_rows()
        )

    def create_attack_report(self, save_path, pdf=False):
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

        def gen_attack_pars_rows(table):
            for key in self.pars_descriptors:
                desc = self.pars_descriptors[key]
                value = str(self.foolbox_attack.__dict__[key])

                table.add_hline()
                table.add_row([desc, value])

        ap = self.attack_pars
        dc = self.data_conf
        self.report_section.append(Subsubsection("Attack Details"))
        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                # -- Create table for the attack parameters.
                self.report_section.append(Command("centering"))
                with self.report_section.create(Tabular("|l|c|")) as tab_ap:
                    tab_ap.add_hline()
                    # Short epsilon array if needed
                    if len(self.epsilons) > 6:
                        eps_str = ""
                        for i in range(3):
                            eps_str = eps_str + str(self.epsilons[i]) + ", "
                        eps_str = eps_str + "..., "
                        for i in range(-3, 0):
                            eps_str = eps_str + str(self.epsilons[i]) + ", "
                        eps_str = eps_str[:-2]
                    else:
                        eps_str = str(self.epsilons)
                        eps_str = str.replace(eps_str, "[", "")
                        eps_str = str.replace(eps_str, "]", "")
                    tab_ap.add_row(["Epsilons", eps_str])
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

        # Epsilon-Misclassification graph
        epsilons = np.array(self.epsilons)
        misclass = np.array(res["success_rate"][tm])
        dist = np.array(res["avg_l2_distance"][tm])

        sort_idx = np.argsort(epsilons)
        fig = plt.figure()
        ax = plt.axes()
        if len(epsilons) > 1:
            ax.plot(epsilons[sort_idx], misclass[sort_idx])
        else:
            ax.plot(epsilons[sort_idx], misclass[sort_idx], "o")
        ax.set_xlabel("Epsilon")
        ax.set_ylabel("Misclassification Rate")
        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        fig.savefig(save_path + f"/fig/{alias_no_spaces}-epsilon_misclass_graph.pdf")
        plt.close(fig)

        with self.report_section.create(MiniPage()):
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))
                self.report_section.append(
                    Command(
                        "includegraphics",
                        NoEscape(f"fig/{alias_no_spaces}-epsilon_misclass_graph.pdf"),
                        "width=8cm",
                    )
                )
                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command(
                        "captionof",
                        "figure",
                        extra_arguments="Epsilon-Misclassification-Rate Graph",
                    )
                )

            # Result table
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))

                if len(epsilons) > 1:
                    worst_epsilon = np.argmin(epsilons)
                    best_epsilon = np.argmax(epsilons)
                    with self.report_section.create(Tabular("|l|c|c|")) as result_tab:
                        result_tab.add_hline()
                        result_tab.add_row(
                            list(map(bold, ["", "Min. Rate", "Max. Rate"]))
                        )
                        result_tab.add_hline()
                        result_tab.add_row(
                            ["Epsilon", epsilons[worst_epsilon], epsilons[best_epsilon]]
                        )
                        result_tab.add_hline()
                        result_tab.add_row(
                            [
                                "Success Rate",
                                round(misclass[worst_epsilon], 3),
                                round(misclass[best_epsilon], 3),
                            ]
                        )
                        result_tab.add_hline()
                        result_tab.add_row(
                            [
                                "Average L2 Distance",
                                round(dist[worst_epsilon], 3),
                                round(dist[best_epsilon], 3),
                            ]
                        )
                        result_tab.add_hline()
                else:
                    with self.report_section.create(Tabular("|l|c|")) as result_tab:
                        result_tab.add_hline()
                        result_tab.add_row(
                            ["Epsilon", epsilons[0]]
                        )
                        result_tab.add_hline()
                        result_tab.add_row(
                            [
                                "Success Rate",
                                round(misclass[0], 3),
                            ]
                        )
                        result_tab.add_hline()
                        result_tab.add_row(
                            [
                                "Average L2 Distance",
                                round(dist[0], 3),
                            ]
                        )
                        result_tab.add_hline()

                self.report_section.append(
                    Command("captionsetup", "labelformat=empty")
                )
                self.report_section.append(
                    Command("captionof", "table", extra_arguments="Attack Summary")
                )


class L2ContrastReductionAttack(BaseAttack):
    """
    foolbox.attacks.L2ContrastReductionAttack wrapper class.

    Attack description:
    Reduces the contrast of the input using a perturbation of the given size.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * target (float): (optional) Target relative to the bounds from 0 (min) to 1
          (max) towards which the contrast is reduced.
        * epsilons (list): List of one or more Epsilons for the attack.

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
        if "target" in attack_pars:
            foolbox_attack = fb.attacks.L2ContrastReductionAttack(
                target=attack_pars["target"]
            )
        else:
            foolbox_attack = fb.attacks.L2ContrastReductionAttack()

        pars_descriptors = {
            "target": "Contrast reduction target",
        }

        super().__init__(
            attack_alias,
            attack_pars["epsilons"],
            data,
            labels,
            data_conf["attack_indices_per_target"],
            target_models,
            foolbox_attack,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "L2 Contrast Reduction Attack",
            self.attack_alias,
            "L2ContrastReductionAttack",
        )


class VirtualAdversarialAttack(BaseAttack):
    """
    foolbox.attacks.VirtualAdversarialAttack wrapper class.

    Attack description:
    Second-order gradient-based attack on the logits. The attack calculate an untargeted
    adversarial perturbation by performing a approximated second order optimization step
    on the KL divergence between the unperturbed predictions and the predictions for the
    adversarial perturbation. This attack was originally introduced as the Virtual
    Adversarial Training method.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * steps (int): Number of update steps.
        * xi (float): (optional) L2 distance between original image and first
          adversarial proposal.
        * epsilons (list): List of one or more Epsilons for the attack.

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
        if "xi" in attack_pars:
            foolbox_attack = fb.attacks.VirtualAdversarialAttack(
                attack_pars["steps"],
                xi=attack_pars["xi"],
            )
        else:
            foolbox_attack = fb.attacks.VirtualAdversarialAttack(attack_pars["steps"])

        pars_descriptors = {
            "steps": "Update steps",
            "xi": "First L2 distance",
        }

        super().__init__(
            attack_alias,
            attack_pars["epsilons"],
            data,
            labels,
            data_conf["attack_indices_per_target"],
            target_models,
            foolbox_attack,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Virtual Adversarial Attack",
            self.attack_alias,
            "VirtualAdversarialAttack",
        )
