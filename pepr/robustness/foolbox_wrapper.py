"""PePR wrapper classes for foolbox attack classes."""

import logging
import numpy as np
import os

from pepr.attack import Attack
from pepr import report
import matplotlib.pyplot as plt
from pylatex import Command, NoEscape, Tabular, MiniPage, Figure
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
        List of one or more epsilons for the attack.
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
        * l2_distance (list): Euclidean distance (L2 norm) between original and
          perturbed images for every single image per target model, epsilon and class
          (shape: (target_models, epsilons, classes, nb_records))
        * avg_l2_distance (list): Average euclidean distance (L2 norm) between original
          and perturbed images (epsilon is upper bound) per target model and epsilon.
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

    def foolbox_run(self, fmodel, inputs_t, criterion_t, epsilons, **kwargs):
        """
        Foolbox attack run function.

        Parameters
        ----------
        fmodel : foolbox.models.Model
            The foolbox model to attack.
        inputs_t : tensorflow.Tensor
            Input tensor.
        criterion_t : tensorflow.Tensor
            True labels or criterion.
        epsilons : iterable
            List of one or more epsilons for the attack.
        kwargs :
            Additional parameters for the `generate` function of the attack.
        """
        return self.foolbox_attack(
            fmodel, inputs_t, criterion_t, epsilons=epsilons, **kwargs
        )

    def run(self, **kwargs):
        """
        Run the Foolbox attack.

        Parameters
        ----------
        kwargs :
            Additional parameters for the `generate` function of the attack.
        """
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
        avg_l2_dist_list = []

        # Run attack for every target model
        for i, fmodel in enumerate(self.fmodels):
            logger.info(f"Attack target model ({i + 1}/{len(self.fmodels)}).")
            indices = self.attack_indices_per_target[i]
            inputs_t = tf.convert_to_tensor(self.data[indices])
            criterion_t = tf.convert_to_tensor(self.labels[indices])

            raw, clipped, is_adv = self.foolbox_run(
                fmodel, inputs_t, criterion_t, epsilons=self.epsilons, **kwargs
            )

            raw_list.append(raw)
            clipped_list.append(clipped)
            is_adv_list.append(is_adv)

            misclass = is_adv.numpy().mean(axis=-1)
            misclass_list.append(misclass)
            logger.debug(f"Average attack accuracy: {misclass}")

            # Calculate average distance of adversarial examples
            dist_eps = []
            avg_dist_eps = []
            for e in range(len(self.epsilons)):
                d = fb.distances.l2(inputs_t, clipped[e]).numpy()
                avg_dist_eps.append(d.mean())
                dist_eps_class = []
                for c in range(np.max(self.labels[indices]) + 1):
                    c_idx, = np.where(self.labels[indices] == c)
                    if c_idx.size == 0:
                        dist_eps_class.append(np.NaN)
                    else:
                        dist_eps_class.append(d[c_idx])
                dist_eps.append(dist_eps_class)
            l2_dist_list.append(dist_eps)
            avg_l2_dist_list.append(avg_dist_eps)

        self.attack_results["raw"] = raw_list
        self.attack_results["clipped"] = clipped_list
        self.attack_results["is_adv"] = is_adv_list
        self.attack_results["success_rate"] = misclass_list
        self.attack_results["l2_distance"] = l2_dist_list
        self.attack_results["avg_l2_distance"] = avg_l2_dist_list

        # Print attack summary
        def _list_to_formatted_string(arr):
            string = ""
            # Short output if needed
            if len(arr) <= 10:
                for item in arr:
                    string = string + f"{round(item, 3):>10}"
            else:
                for t in range(5):
                    string = string + f"{round(arr[t], 3):>10}"
                string = string + f"{'...':>10}"
                for t in range(-5, 0):
                    string = string + f"{round(arr[t], 3):>10}"
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

    def create_attack_report(self, save_path="foolbox_report", pdf=False):
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
                if key == "distance":
                    key = "_distance"
                value = str(self.foolbox_attack.__dict__[key])

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
                    # Short epsilon array if needed
                    eps = np.round(self.epsilons, decimals=3)
                    if len(eps) > 6:
                        eps_str = ""
                        for i in range(3):
                            eps_str = eps_str + str(eps[i]) + ", "
                        eps_str = eps_str + "..., "
                        for i in range(-3, 0):
                            eps_str = eps_str + str(eps[i]) + ", "
                        eps_str = eps_str[:-2]
                    else:
                        eps_str = str(eps)
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

    def _plot_most_vulnerable_aes(self, save_path, target_model_index, count):
        """
        Plot most vulnerable (real) adversarial examples.
        Parameters
        ----------
        self : BaseEvasionAttack or BasePatchAttack
            Base attack object.
        target_model_index : int
            Index of the target model of the adversarial examples.
        count : int
            Number of adversarial examples to display. If `count` lower or equal than
            the number of classes, the most vulnerable images of the most vulnerable
            classes are plotted (0-1 per class). If `count` greater than the number of
            classes, starting by the class with the most vulnerable image one image is
            plotted until `count` images were plotted (`count/nb_classes` to
            `count/nb_classes + 1` per class).
        """

        def argsort_by_nth_element(arr, n):
            nth = []
            nan_count = 0
            for i in range(len(arr)):
                if arr[i] is np.NaN or len(arr[i]) <= n:
                    nth.append(np.NaN)
                    nan_count = nan_count + 1
                else:
                    nth.append(arr[i][n])

            if nan_count == 0:
                return np.argsort(nth)
            else:
                return np.argsort(nth)[:-nan_count]

        max_rate_eps = np.argmax(
            self.attack_results["success_rate"][target_model_index]
        )

        org_data = self.data[self.attack_indices_per_target[target_model_index]]
        adv_data = self.attack_results["clipped"][target_model_index]
        labels = self.labels[self.attack_indices_per_target[target_model_index]]
        is_adv = self.attack_results["is_adv"][target_model_index]

        adv_data = adv_data[max_rate_eps].numpy()
        is_adv = is_adv[max_rate_eps].numpy()

        nb_classes = np.max(labels) + 1
        nb_adv = 0

        tmp_is_adv = []
        for c in range(nb_classes):
            indices = np.where(labels == c)
            tmp_is_adv.append(is_adv[indices])
        is_adv = tmp_is_adv

        data = []
        adv = []
        dists = []
        true_labels = []
        predicted_labels = []

        for l in range(nb_classes):
            if is_adv[l] is np.NaN or np.count_nonzero(is_adv[l]) == 0:
                data.append(np.NaN)
                adv.append(np.NaN)
                dists.append(np.NaN)
                true_labels.append(np.NaN)
                predicted_labels.append(np.NaN)
                continue
            indices = np.extract(is_adv[l], np.where(labels == l))
            class_data = org_data[indices]
            class_adv_data = adv_data[indices]
            class_labels = labels[indices]
            class_dist = np.extract(
                is_adv[l],
                self.attack_results["l2_distance"][target_model_index][max_rate_eps][l],
            )
            sort_idxs = np.argsort(class_dist)

            pred = self.target_models[target_model_index].predict(class_adv_data)

            data.append(class_data[sort_idxs])
            adv.append(class_adv_data[sort_idxs])
            dists.append(class_dist[sort_idxs])
            true_labels.append(class_labels[sort_idxs])
            predicted_labels.append(pred[sort_idxs])
            nb_adv = nb_adv + len(sort_idxs)

        idx = 0
        plot_count = 0
        ncols = min(count, nb_adv)
        fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(15, 5))
        while plot_count < count:
            cls = argsort_by_nth_element(dists, idx)
            if len(cls) == 0:
                break
            for c in cls:
                if len(adv[c]) > idx:
                    image_org = data[c][idx]
                    image_adv = adv[c][idx]
                    logger.debug(f"Image: {plot_count}")
                    logger.debug(f"True Label: {true_labels[c][idx]}")
                    logger.debug(
                        f"Prediction Label: {np.argmax(predicted_labels[c][idx])}"
                    )
                    logger.debug(f"Distance: {dists[c][idx]}")
                    # Plot original
                    ax = axes[0][plot_count]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(f"Original: {true_labels[c][idx]}")
                    if image_org.shape[-1] == 1:
                        ax.imshow(image_org[:, :, 0])
                    else:
                        ax.imshow(image_org)
                    # Plot adversarial
                    ax = axes[1][plot_count]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xlabel(
                        f"Predicted: {np.argmax(predicted_labels[c][idx])}\n"
                        + f"Dist.: {str(round(dists[c][idx], 3))}"
                    )
                    if image_adv.shape[-1] == 1:
                        ax.imshow(image_adv[:, :, 0])
                    else:
                        ax.imshow(image_adv)
                    plot_count = plot_count + 1
                    if plot_count >= count:
                        break
            idx = idx + 1

        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        fig.savefig(
            save_path + f"/fig/{alias_no_spaces}-examples.pdf", bbox_inches="tight"
        )
        plt.close(fig)

    def _report_attack_results(self, save_path):
        """
        Create subsubsection describing the most important results of the attack.

        Note: If only one epsilon is given, a histogram is generated showing the
        distribution of the success rate over the classes. If multiple epsilons are
        given, a Epsilon-Misclassification is generated.

        Parameters
        ----------
        save_path :
            Path to save the tex, pdf and asset files of the attack report.

        This subsection contains results only for the first target model.
        """
        tm = 0  # Specify target model
        self.report_section.append(Subsubsection("Attack Results"))
        res = self.attack_results

        alias_no_spaces = str.replace(self.attack_alias, " ", "_")
        epsilons = np.array(self.epsilons)
        misclass = np.array(res["success_rate"][tm])
        dist = np.array(res["avg_l2_distance"][tm])

        if len(epsilons) > 1:
            # Epsilon-Misclassification graph
            title = "Epsilon-Misclassification-Rate"
            path = f"fig/{alias_no_spaces}-epsilon_misclass_graph.pdf"

            sort_idx = np.argsort(epsilons)
            fig = plt.figure()
            ax = plt.axes()
            if len(epsilons) > 1:
                ax.plot(epsilons[sort_idx], misclass[sort_idx])
            else:
                ax.plot(epsilons[sort_idx], misclass[sort_idx], "o")
            ax.set_xlabel("Epsilon")
            ax.set_ylabel("Misclassification Rate")
            fig.savefig(save_path + f"/{path}")
            plt.close(fig)
        else:
            # Histogram
            misclass_class = []
            labels = self.labels[self.attack_indices_per_target[tm]]
            is_adv_eps = self.attack_results["is_adv"][tm][0].numpy()
            for c in range(np.max(labels) + 1):
                class_idx = np.where(labels == c)
                misclass_class.append(np.mean(is_adv_eps[class_idx]))

            fig = plt.figure()
            ax = plt.axes()
            ax.hist(misclass_class, edgecolor="black")
            ax.set_xlabel("Accuracy")
            ax.set_ylabel("Number of Classes")
            ax.set_axisbelow(True)

            path = f"fig/{alias_no_spaces}-hist.pdf"
            title = "Success Rate Distribution"
            fig.savefig(save_path + f"/{path}")
            plt.close(fig)

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
                        extra_arguments=title,
                    )
                )

            # Result table
            with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
                self.report_section.append(Command("centering"))

                if len(epsilons) > 1:
                    worst_epsilon = np.argmin(misclass)
                    best_epsilon = np.argmax(misclass)
                    with self.report_section.create(Tabular("|l|c|c|")) as result_tab:
                        result_tab.add_hline()
                        result_tab.add_row(
                            list(map(bold, ["", "Min. Rate", "Max. Rate"]))
                        )
                        result_tab.add_hline()
                        result_tab.add_row(
                            [
                                "Epsilon",
                                round(epsilons[worst_epsilon], 4),
                                round(epsilons[best_epsilon], 4),
                            ]
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
                        result_tab.add_row(["Epsilon", epsilons[0]])
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

                self.report_section.append(Command("captionsetup", "labelformat=empty"))
                self.report_section.append(
                    Command("captionof", "table", extra_arguments="Attack Summary")
                )

        self._plot_most_vulnerable_aes(save_path, tm, 10)
        with self.report_section.create(Figure(position="H")) as fig:
            fig.add_image(
                f"fig/{alias_no_spaces}-examples.pdf", width=NoEscape(r"\textwidth")
            )
            self.report_section.append(Command("captionsetup", "labelformat=empty"))
            self.report_section.append(
                Command(
                    "captionof",
                    "figure",
                    extra_arguments="This is a small selection of the most vulnerable "
                    "adversarial examples per class and of the epsilon with the "
                    "highest success rate. They were sorted per class by the lowest "
                    "distance which changes the target's prediction. Sorting per "
                    "class for every n-th image may not give the absolute most "
                    "vulnerable records but provides the highest diversity. "
                    "(First row: Originals, second row: Adversarial examples)",
                )
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
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "target": "Contrast reduction target",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2ContrastReductionAttack(**params)

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
            "FB_L2ContrastReductionAttack",
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
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "steps": "Update steps",
            "xi": "First L2 distance",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.VirtualAdversarialAttack(**params)

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
            "FB_VirtualAdversarialAttack",
        )


class DDNAttack(BaseAttack):
    """
    foolbox.attacks.DDNAttack wrapper class.

    Attack description:
    The Decoupled Direction and Norm L2 adversarial attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * init_epsilon (float): (optional) Initial value for the norm/epsilon ball.
        * steps (int): (optional) Number of steps for the optimization.
        * gamma (float): (optional) Factor by which the norm will be modified:
          new_norm = norm * (1 + or - gamma).
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "init_epsilon": "Initial epsilon ball",
            "steps": "Optimization steps",
            "gamma": "Norm factor",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.DDNAttack(**params)

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
            "DDN Attack",
            self.attack_alias,
            "FB_DDNAttack",
        )


class L2ProjectedGradientDescentAttack(BaseAttack):
    """
    foolbox.attacks.L2ProjectedGradientDescentAttack wrapper class.

    Attack description:
    L2 Projected Gradient Descent.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * rel_stepsize (float): (optional) Stepsize relative to epsilon.
        * abs_stepsize (float): (optional) If given, it takes precedence over
          rel_stepsize.
        * steps (int): (optional) Number of update steps to perform.
        * random_start (bool): (optional) Whether the perturbation is initialized
          randomly or starts at zero.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "rel_stepsize": "Relative stepsize",
            "abs_stepsize": "Absolut stepsize",
            "steps": "Update steps",
            "random_start": "Random start",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2ProjectedGradientDescentAttack(**params)

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
            "L2 Projected Gradient Descent Attack",
            self.attack_alias,
            "FB_L2ProjectedGradientDescentAttack",
        )


class LinfProjectedGradientDescentAttack(BaseAttack):
    """
    foolbox.attacks.LinfProjectedGradientDescentAttack wrapper class.

    Attack description:
    Linf Projected Gradient Descent.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * rel_stepsize (float): (optional) Stepsize relative to epsilon.
        * abs_stepsize (float): (optional) If given, it takes precedence over
          rel_stepsize.
        * steps (int): (optional) Number of update steps to perform.
        * random_start (bool): (optional) Whether the perturbation is initialized
          randomly or starts at zero.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "rel_stepsize": "Relative stepsize",
            "abs_stepsize": "Absolut stepsize",
            "steps": "Update steps",
            "random_start": "Random start",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinfProjectedGradientDescentAttack(**params)

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
            "Linf Projected Gradient Descent Attack",
            self.attack_alias,
            "FB_LinfProjectedGradientDescentAttack",
        )


class L2BasicIterativeAttack(BaseAttack):
    """
    foolbox.attacks.L2BasicIterativeAttack wrapper class.

    Attack description:
    L2 Basic Iterative Method.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * rel_stepsize (float): (optional) Stepsize relative to epsilon.
        * abs_stepsize (float): (optional) If given, it takes precedence over
          rel_stepsize.
        * steps (int): (optional) Number of update steps.
        * random_start (bool): (optional) Controls whether to randomly start within
          allowed epsilon ball.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "rel_stepsize": "Relative stepsize",
            "abs_stepsize": "Absolut stepsize",
            "steps": "Update steps",
            "random_start": "Random start",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2BasicIterativeAttack(**params)

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
            "L2 Basic Iterative Attack",
            self.attack_alias,
            "FB_L2BasicIterativeAttack",
        )


class LinfBasicIterativeAttack(BaseAttack):
    """
    foolbox.attacks.LinfBasicIterativeAttack wrapper class.

    Attack description:
    L-infinity Basic Iterative Method.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * rel_stepsize (float): (optional) Stepsize relative to epsilon.
        * abs_stepsize (float): (optional) If given, it takes precedence over
          rel_stepsize.
        * steps (int): (optional) Number of update steps.
        * random_start (bool): (optional) Controls whether to randomly start within
          allowed epsilon ball.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "rel_stepsize": "Relative stepsize",
            "abs_stepsize": "Absolut stepsize",
            "steps": "Update steps",
            "random_start": "Random start",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinfBasicIterativeAttack(**params)

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
            "Linf Basic Iterative Attack",
            self.attack_alias,
            "FB_LinfBasicIterativeAttack",
        )


class L2FastGradientAttack(BaseAttack):
    """
    foolbox.attacks.L2FastGradientAttack wrapper class.

    Attack description:
    Fast Gradient Method (FGM).

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * random_start (bool): (optional) Controls whether to randomly start within
          allowed epsilon ball.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "random_start": "Random start",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2FastGradientAttack(**params)

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
            "L2 Fast Gradient Attack",
            self.attack_alias,
            "FB_L2FastGradientAttack",
        )


class LinfFastGradientAttack(BaseAttack):
    """
    foolbox.attacks.LinfFastGradientAttack wrapper class.

    Attack description:
    Fast Gradient Sign Method (FGSM).

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * random_start (bool): (optional) Controls whether to randomly start within
          allowed epsilon ball.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "random_start": "Random start",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinfFastGradientAttack(**params)

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
            "Linf Fast Gradient Attack",
            self.attack_alias,
            "FB_LinfFastGradientAttack",
        )


class L2AdditiveGaussianNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2AdditiveGaussianNoiseAttack wrapper class.

    Attack description:
    Samples Gaussian noise with a fixed L2 size.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * epsilons (list): List of one or more epsilons for the attack.

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
        foolbox_attack = fb.attacks.L2AdditiveGaussianNoiseAttack()

        pars_descriptors = {}

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
            "L2 Additive Gaussian Noise Attack",
            self.attack_alias,
            "FB_L2AdditiveGaussianNoiseAttack",
        )


class L2AdditiveUniformNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2AdditiveUniformNoiseAttack wrapper class.

    Attack description:
    Samples uniform noise with a fixed L2 size.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * epsilons (list): List of one or more epsilons for the attack.

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
        foolbox_attack = fb.attacks.L2AdditiveUniformNoiseAttack()

        pars_descriptors = {}

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
            "L2 Additive Uniform Noise Attack",
            self.attack_alias,
            "FB_L2AdditiveUniformNoiseAttack",
        )


class L2ClippingAwareAdditiveGaussianNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack wrapper class.

    Attack description:
    Samples Gaussian noise with a fixed L2 size after clipping.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * epsilons (list): List of one or more epsilons for the attack.

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
        foolbox_attack = fb.attacks.L2ClippingAwareAdditiveGaussianNoiseAttack()

        pars_descriptors = {}

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
            "L2 Clipping Aware Additive Gaussian Noise Attack",
            self.attack_alias,
            "FB_L2ClippingAwareAdditiveGaussianNoiseAttack",
        )


class L2ClippingAwareAdditiveUniformNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2ClippingAwareAdditiveUniformNoiseAttack wrapper class.

    Attack description:
    Samples uniform noise with a fixed L2 size after clipping.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * epsilons (list): List of one or more epsilons for the attack.

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
        foolbox_attack = fb.attacks.L2ClippingAwareAdditiveUniformNoiseAttack()

        pars_descriptors = {}

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
            "L2 Clipping Aware Additive Uniform Noise Attack",
            self.attack_alias,
            "FB_L2ClippingAwareAdditiveUniformNoiseAttack",
        )


class LinfAdditiveUniformNoiseAttack(BaseAttack):
    """
    foolbox.attacks.LinfAdditiveUniformNoiseAttack wrapper class.

    Attack description:
    Samples uniform noise with a fixed L-infinity size.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * epsilons (list): List of one or more epsilons for the attack.

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
        foolbox_attack = fb.attacks.LinfAdditiveUniformNoiseAttack()

        pars_descriptors = {}

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
            "Linf Additive Uniform Noise Attack",
            self.attack_alias,
            "FB_LinfAdditiveUniformNoiseAttack",
        )


class L2RepeatedAdditiveGaussianNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2RepeatedAdditiveGaussianNoiseAttack wrapper class.

    Attack description:
    Repeatedly samples Gaussian noise with a fixed L2 size.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * repeats (int): (optional) How often to sample random noise.
        * check_trivial (bool): (optional) Check whether original sample is already
          adversarial.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "repeats": "Repeats",
            "check_trivial": "Check adversarial",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2RepeatedAdditiveGaussianNoiseAttack(**params)

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
            "L2 Repeated Additive Gaussian Noise Attack",
            self.attack_alias,
            "FB_L2RepeatedAdditiveGaussianNoiseAttack",
        )


class L2RepeatedAdditiveUniformNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2RepeatedAdditiveUniformNoiseAttack wrapper class.

    Attack description:
    Repeatedly samples uniform noise with a fixed L2 size.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * repeats (int): (optional) How often to sample random noise.
        * check_trivial (bool): (optional) Check whether original sample is already
          adversarial.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "repeats": "Repeats",
            "check_trivial": "Check adversarial",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2RepeatedAdditiveUniformNoiseAttack(**params)

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
            "L2 Repeated Additive Uniform Noise Attack",
            self.attack_alias,
            "FB_L2RepeatedAdditiveUniformNoiseAttack",
        )


class L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack wrapper class.

    Attack description:
    Repeatedly samples Gaussian noise with a fixed L2 size after clipping.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * repeats (int): (optional) How often to sample random noise.
        * check_trivial (bool): (optional) Check whether original sample is already
          adversarial.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "repeats": "Repeats",
            "check_trivial": "Check adversarial",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack(
            **params
        )

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
            "L2 Clipping Aware Repeated Additive Gaussian Noise Attack",
            self.attack_alias,
            "FB_L2ClippingAwareRepeatedAdditiveGaussianNoiseAttack",
        )


class L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(BaseAttack):
    """
    foolbox.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack wrapper class.

    Attack description:
    Repeatedly samples uniform noise with a fixed L2 size after clipping.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * repeats (int): (optional) How often to sample random noise.
        * check_trivial (bool): (optional) Check whether original sample is already
          adversarial.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "repeats": "Repeats",
            "check_trivial": "Check adversarial",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2ClippingAwareRepeatedAdditiveUniformNoiseAttack(
            **params
        )

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
            "L2 Clipping Aware Repeated Additive Uniform Noise Attack",
            self.attack_alias,
            "FB_L2ClippingAwareRepeatedAdditiveUniformNoiseAttack",
        )


class LinfRepeatedAdditiveUniformNoiseAttack(BaseAttack):
    """
    foolbox.attacks.LinfRepeatedAdditiveUniformNoiseAttack wrapper class.

    Attack description:
    Repeatedly samples uniform noise with a fixed L-infinity size.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * repeats (int): (optional) How often to sample random noise.
        * check_trivial (bool): (optional) Check whether original sample is already
          adversarial.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "repeats": "Repeats",
            "check_trivial": "Check adversarial",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinfRepeatedAdditiveUniformNoiseAttack(**params)

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
            "Linf Repeated Additive Uniform Noise Attack",
            self.attack_alias,
            "FB_LinfRepeatedAdditiveUniformNoiseAttack",
        )


class InversionAttack(BaseAttack):
    """
    foolbox.attacks.InversionAttack wrapper class.

    Attack description:
    Creates "negative images" by inverting the pixel values.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * distance (foolbox.distances.Distance): Distance measure for which minimal
          adversarial examples are searched.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "distance": "Distance",
        }

        foolbox_attack = fb.attacks.InversionAttack(distance=attack_pars["distance"])

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
            "Inversion Attack",
            self.attack_alias,
            "FB_InversionAttack",
        )


class BinarySearchContrastReductionAttack(BaseAttack):
    """
    foolbox.attacks.BinarySearchContrastReductionAttack wrapper class.

    Attack description:
    Reduces the contrast of the input using a binary search to find the smallest
    adversarial perturbation

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * distance (foolbox.distances.Distance): Distance measure for which minimal
          adversarial examples are searched.
        * binary_search_steps (int): (optional) Number of iterations in the binary
          search. This controls the precision of the results.
        * target (float): (optional) Target relative to the bounds from 0 (min) to 1
          (max) towards which the contrast is reduced.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "distance": "Distance",
            "binary_search_steps": "Binary search iterations",
            "target": "Contrast target",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.BinarySearchContrastReductionAttack(**params)

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
            "Binary Search Contrast Reduction Attack",
            self.attack_alias,
            "FB_BinarySearchContrastReductionAttack",
        )


class LinearSearchContrastReductionAttack(BaseAttack):
    """
    foolbox.attacks.LinearSearchContrastReductionAttack wrapper class.

    Attack description:
    Reduces the contrast of the input using a linear search to find the smallest
    adversarial perturbation.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * distance (foolbox.distances.Distance): Distance measure for which minimal
          adversarial examples are searched.
        * steps (int): (optional) Number of iterations in the linear search. This
          controls the precision of the results.
        * target (float): (optional) Target relative to the bounds from 0 (min) to 1
          (max) towards which the contrast is reduced.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "distance": "Distance",
            "steps": "Linear search iterations",
            "target": "Contrast target",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinearSearchContrastReductionAttack(**params)

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
            "Linear Search Contrast Reduction Attack",
            self.attack_alias,
            "FB_LinearSearchContrastReductionAttack",
        )


class L2CarliniWagnerAttack(BaseAttack):
    """
    foolbox.attacks.L2CarliniWagnerAttack wrapper class.

    Attack description:
    Implementation of the Carlini & Wagner L2 Attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * binary_search_steps (int): (optional) Number of steps to perform in the binary
          search over the const c.
        * steps (int): (optional) Number of optimization steps within each binary search
          step.
        * stepsize (float): (optional) Stepsize to update the examples.
        * confidence (float): (optional) Confidence required for an example to be marked
          as adversarial. Controls the gap between example and decision boundary.
        * initial_const (float): (optional) Initial value of the const c with which the
          binary search starts.
        * abort_early (bool): (optional) Stop inner search as soon as an adversarial
          example has been found. Does not affect the binary search over the const c.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "binary_search_steps": "Binary search iterations",
            "steps": "Optimization steps",
            "stepsize": "Stepsize",
            "confidence": "Confidence marking adversarial",
            "initial_const": "Initial cost value",
            "abort_early": "Stop early",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2CarliniWagnerAttack(**params)

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
            "L2 Carlini Wagner Attack",
            self.attack_alias,
            "FB_L2CarliniWagnerAttack",
        )


class NewtonFoolAttack(BaseAttack):
    """
    foolbox.attacks.NewtonFoolAttack wrapper class.

    Attack description:
    Implementation of the NewtonFool Attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * steps (int): (optional) Number of update steps to perform..
        * stepsize (float): (optional) Size of each update step..
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "steps": "Update steps",
            "stepsize": "Stepsize",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.NewtonFoolAttack(**params)

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
            "Newton Fool Attack",
            self.attack_alias,
            "FB_NewtonFoolAttack",
        )


class EADAttack(BaseAttack):
    """
    foolbox.attacks.EADAttack wrapper class.

    Attack description:
    Implementation of the EAD Attack with EN Decision Rule.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * binary_search_steps (int): (optional) Number of steps to perform in the binary
          search over the const c.
        * steps (int): (optional) Number of optimization steps within each binary search
          step.
        * initial_stepsize (float): (optional) Initial stepsize to update the examples.
        * confidence (float): (optional) Confidence required for an example to be marked
          as adversarial. Controls the gap between example and decision boundary.
        * initial_const (float): (optional) Initial value of the const c with which the
          binary search starts.
        * regularization (float): (optional) Controls the L1 regularization.
        * decision_rule ("EN" ir "L1"): (optional) Rule according to which the best
          adversarial examples are selected. They either minimize the L1 or ElasticNet
          distance.
        * abort_early (bool): (optional) Stop inner search as soon as an adversarial
          example has been found. Does not affect the binary search over the const c.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "binary_search_steps": "Binary search iterations",
            "steps": "Optimization steps",
            "initial_stepsize": "Stepsize",
            "confidence": "Confidence marking adversarial",
            "initial_const": "Initial cost value",
            "regularization": "L1 regularization",
            "decision_rule": "Decision Rule",
            "abort_early": "Stop early",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.EADAttack(**params)

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
            "EAD Attack",
            self.attack_alias,
            "FB_EADAttack",
        )


class GaussianBlurAttack(BaseAttack):
    """
    foolbox.attacks.GaussianBlurAttack wrapper class.

    Attack description:
    Blurs the inputs using a Gaussian filter with linearly increasing standard
    deviation.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * steps (int): (optional) Number of sigma values tested between 0 and max_sigma.
        * channel_axis (int): (optional) Index of the channel axis in the input data.
        * max_sigma (float): (optional) Maximally allowed sigma value of the Gaussian
          blur.
        * distance (foolbox.distances.Distance): Distance measure for which minimal
          adversarial examples are searched.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "steps": "Number sigma values",
            "channel_axis": "Channel Axis",
            "max_sigma": "Max. sigma",
            "distance": "Distance",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.GaussianBlurAttack(**params)

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
            "Gaussian Blur Attack",
            self.attack_alias,
            "FB_GaussianBlurAttack",
        )


class L2DeepFoolAttack(BaseAttack):
    """
    foolbox.attacks.L2DeepFoolAttack wrapper class.

    Attack description:
    A simple and fast gradient-based adversarial attack. Implements the DeepFool L2
    attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * steps (int): (optional) Maximum number of steps to perform.
        * candidates (int): (optional) Limit on the number of the most likely classes
          that should be considered. A small value is usually sufficient and much
          faster.
        * overshoot (float): (optional) How much to overshoot the boundary.
        * loss ("crossentropy" or "logits"): (optional) Loss function to use inside the
          update function.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "steps": "Maximum steps",
            "candidates": "Max. candidates",
            "overshoot": "Overshoot",
            "loss": "Loss function",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2DeepFoolAttack(**params)

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
            "L2 Deep Fool Attack",
            self.attack_alias,
            "FB_L2DeepFoolAttack",
        )


class LinfDeepFoolAttack(BaseAttack):
    """
    foolbox.attacks.LinfDeepFoolAttack wrapper class.

    Attack description:
    A simple and fast gradient-based adversarial attack. Implements the DeepFool
    L-Infinity attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * steps (int): (optional) Maximum number of steps to perform.
        * candidates (int): (optional) Limit on the number of the most likely classes
          that should be considered. A small value is usually sufficient and much
          faster.
        * overshoot (float): (optional) How much to overshoot the boundary.
        * loss ("crossentropy" or "logits"): (optional) Loss function to use inside the
          update function.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "steps": "Maximum steps",
            "candidates": "Max. candidates",
            "overshoot": "Overshoot",
            "loss": "Loss function",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinfDeepFoolAttack(**params)

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
            "Linf Deep Fool Attack",
            self.attack_alias,
            "FB_LinfDeepFoolAttack",
        )


class SaltAndPepperNoiseAttack(BaseAttack):
    """
    foolbox.attacks.SaltAndPepperNoiseAttack wrapper class.

    Attack description:
    Increases the amount of salt and pepper noise until the input is misclassified.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * steps (int): (optional) The number of steps to run.
        * across_channels (bool): Whether the noise should be the same across all
          channels.
        * channel_axis (int): (optional) The axis across which the noise should be the
          same (if across_channels is True). If None, will be automatically inferred
          from the model if possible.
        * epsilons (list): (optional) List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "steps": "Steps",
            "across_channels": "Same noise",
            "channel_axis": "Same noise channel",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.SaltAndPepperNoiseAttack(**params)

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
            "Salt And Pepper Noise Attack",
            self.attack_alias,
            "FB_SaltAndPepperNoiseAttack",
        )


class LinearSearchBlendedUniformNoiseAttack(BaseAttack):
    """
    foolbox.attacks.LinearSearchBlendedUniformNoiseAttack wrapper class.

    Attack description:
    Blends the input with a uniform noise input until it is misclassified.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * distance (foolbox.distances.Distance): Distance measure for which minimal
          adversarial examples are searched.
        * directions (int): (optional) Number of random directions in which the
          perturbation is searched.
        * steps (int): (optional) Number of blending steps between the original image
          and the random directions.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "distance": "Distance",
            "directions": "Random directions",
            "steps": "Steps",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinearSearchBlendedUniformNoiseAttack(**params)

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
            "Linear Search Blended Uniform Noise Attack",
            self.attack_alias,
            "FB_LinearSearchBlendedUniformNoiseAttack",
        )


class BinarizationRefinementAttack(BaseAttack):
    """
    foolbox.attacks.BinarizationRefinementAttack wrapper class.

    Attack description:
    For models that preprocess their inputs by binarizing the inputs, this attack can
    improve adversarials found by other attacks. It does this by utilizing information
    about the binarization and mapping values to the corresponding value in the clean
    input or to the right side of the threshold.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * starting_points (list): Adversarial examples to improve.
        * threshold (float): (optional) The threshold used by the models binarization.
          If none, defaults to (model.bounds()[1] - model.bounds()[0]) / 2.
        * included_in ("lower" or "upper"): (optional) Whether the threshold value
          itself belongs to the lower or upper interval.
        * distance (foolbox.distances.Distance): Distance measure for which minimal
          adversarial examples are searched.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "threshold": "Threshold",
            "included_in": "Threshold belonging",
            "distance": "Distance",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.BinarizationRefinementAttack(**params)

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
            "Binarization Refinement Attack",
            self.attack_alias,
            "FB_BinarizationRefinementAttack",
        )

        self.starting_points = attack_pars["starting_points"]

    def foolbox_run(self, fmodel, inputs_t, criterion_t, epsilons, **kwargs):
        """
        Run Foolbox attack with starting points.

        Parameters
        ----------
        fmodel : foolbox.models.Model
            The foolbox model to attack.
        inputs_t : tensorflow.Tensor
            Input tensor.
        criterion_t : tensorflow.Tensor
            True labels or criterion.
        epsilons : iterable
            List of one or more epsilons for the attack.
        """
        return self.foolbox_attack(
            fmodel,
            inputs_t,
            criterion_t,
            epsilons=epsilons,
            starting_points=self.starting_points,
        )


class BoundaryAttack(BaseAttack):
    """
    foolbox.attacks.BoundaryAttack wrapper class.

    Attack description:
    A powerful adversarial attack that requires neither gradients nor probabilities.
    This is the reference implementation for the attack.

    **Notes**
    Differences to the original reference implementation:

    * We do not perform internal operations with float64
    * The samples within a batch can currently influence each other a bit
    * We don’t perform the additional convergence confirmation
    * The success rate tracking changed a bit
    * Some other changes due to batching and merged loops

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * init_attack (Optional[foolbox.attacks.base.MinimizationAttack]): (optional)
          Attack to use to find a starting points. Defaults to
          LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        * steps (int): Maximum number of steps to run. Might converge and stop before
          that.
        * spherical_step (float): (optional) Initial step size for the orthogonal
          (spherical) step.
        * source_step (float): (optional) Initial step size for the step towards the
          target.
        * source_step_convergence (float): (optional) Sets the threshold of the stop
          criterion: if source_step becomes smaller than this value during the attack,
          the attack has converged and will stop.
        * step_adaptation (float): (optional) Factor by which the step sizes are
          multiplied or divided.
        * tensorboard (Union[typing_extensions.Literal[False], None, str]): (optional)
          The log directory for TensorBoard summaries. If False, TensorBoard summaries
          will be disabled (default). If None, the logdir will be
          runs/CURRENT_DATETIME_HOSTNAME.
        * update_stats_every_k (int): (optional)
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "init_attack": "Starting attack",
            "steps": "Maximum steps",
            "spherical_step": "Orthogonal step",
            "source_step": "Stepn size",
            "source_step_convergence": "Stop criterion threshold",
            "step_adaptation": "Step factor",
            "tensorboard": "TensorBoard summary",
            "update_stats_every_k": "Stats update",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.BoundaryAttack(**params)

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
            "Boundary Attack",
            self.attack_alias,
            "FB_BoundaryAttack",
        )


class L0BrendelBethgeAttack(BaseAttack):
    """
    foolbox.attacks.L0BrendelBethgeAttack wrapper class.

    Attack description:
    L0 variant of the Brendel & Bethge adversarial attack. This is a powerful
    gradient-based adversarial attack that follows the adversarial boundary (the
    boundary between the space of adversarial and non-adversarial images as defined by
    the adversarial criterion) to find the minimum distance to the clean image.

    This is the reference implementation of the Brendel & Bethge attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * init_attack (Optional[foolbox.attacks.base.MinimizationAttack]): (optional)
          Attack to use to find a starting points. Defaults to
          LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        * overshoot (float): (optional)
        * steps (int): (optional) Maximum number of steps to run.
        * lr (float): (optional)
        * lr_decay (float): (optional)
        * lr_num_decay (int): (optional)
        * momentum (float): (optional)
        * tensorboard (Union[typing_extensions.Literal[False], None, str]): (optional)
          The log directory for TensorBoard summaries. If False, TensorBoard summaries
          will be disabled (default). If None, the logdir will be
          runs/CURRENT_DATETIME_HOSTNAME.
        * binary_search_steps (int): (optional) Number of iterations in the binary
          search. This controls the precision of the results.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "init_attack": "Starting attack",
            "overshoot": "overshoot",
            "steps": "Maximum steps",
            "lr": "LR",
            "lr_decay": "LR decay",
            "lr_num_decay": "LR decay number",
            "momentum": "Momentum",
            "tensorboard": "TensorBoard summary",
            "binary_search_steps": "Binary search iterations",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L0BrendelBethgeAttack(**params)

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
            "L0 Brendel Bethge Attack",
            self.attack_alias,
            "FB_L0BrendelBethgeAttack",
        )

    def __getstate__(self):
        del self.foolbox_attack.__dict__["_optimizer"]
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.foolbox_attack._optimizer = fb.attacks.brendel_bethge.L0Optimizer()


class L1BrendelBethgeAttack(BaseAttack):
    """
    foolbox.attacks.L1BrendelBethgeAttack wrapper class.

    Attack description:
    L1 variant of the Brendel & Bethge adversarial attack. This is a powerful
    gradient-based adversarial attack that follows the adversarial boundary (the
    boundary between the space of adversarial and non-adversarial images as defined by
    the adversarial criterion) to find the minimum distance to the clean image.

    This is the reference implementation of the Brendel & Bethge attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * init_attack (Optional[foolbox.attacks.base.MinimizationAttack]): (optional)
          Attack to use to find a starting points. Defaults to
          LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        * overshoot (float): (optional)
        * steps (int): (optional) Maximum number of steps to run.
        * lr (float): (optional)
        * lr_decay (float): (optional)
        * lr_num_decay (int): (optional)
        * momentum (float): (optional)
        * tensorboard (Union[typing_extensions.Literal[False], None, str]): (optional)
          The log directory for TensorBoard summaries. If False, TensorBoard summaries
          will be disabled (default). If None, the logdir will be
          runs/CURRENT_DATETIME_HOSTNAME.
        * binary_search_steps (int): (optional) Number of iterations in the binary
          search. This controls the precision of the results.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "init_attack": "Starting attack",
            "overshoot": "overshoot",
            "steps": "Maximum steps",
            "lr": "LR",
            "lr_decay": "LR decay",
            "lr_num_decay": "LR decay number",
            "momentum": "Momentum",
            "tensorboard": "TensorBoard summary",
            "binary_search_steps": "Binary search iterations",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L1BrendelBethgeAttack(**params)

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
            "L1 Brendel Bethge Attack",
            self.attack_alias,
            "FB_L1BrendelBethgeAttack",
        )

    def __getstate__(self):
        del self.foolbox_attack.__dict__["_optimizer"]
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.foolbox_attack._optimizer = fb.attacks.brendel_bethge.L1Optimizer()


class L2BrendelBethgeAttack(BaseAttack):
    """
    foolbox.attacks.L2BrendelBethgeAttack wrapper class.

    Attack description:
    L2 variant of the Brendel & Bethge adversarial attack. This is a powerful
    gradient-based adversarial attack that follows the adversarial boundary (the
    boundary between the space of adversarial and non-adversarial images as defined by
    the adversarial criterion) to find the minimum distance to the clean image.

    This is the reference implementation of the Brendel & Bethge attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * init_attack (Optional[foolbox.attacks.base.MinimizationAttack]): (optional)
          Attack to use to find a starting points. Defaults to
          LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        * overshoot (float): (optional)
        * steps (int): (optional) Maximum number of steps to run.
        * lr (float): (optional)
        * lr_decay (float): (optional)
        * lr_num_decay (int): (optional)
        * momentum (float): (optional)
        * tensorboard (Union[typing_extensions.Literal[False], None, str]): (optional)
          The log directory for TensorBoard summaries. If False, TensorBoard summaries
          will be disabled (default). If None, the logdir will be
          runs/CURRENT_DATETIME_HOSTNAME.
        * binary_search_steps (int): (optional) Number of iterations in the binary
          search. This controls the precision of the results.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "init_attack": "Starting attack",
            "overshoot": "overshoot",
            "steps": "Maximum steps",
            "lr": "LR",
            "lr_decay": "LR decay",
            "lr_num_decay": "LR decay number",
            "momentum": "Momentum",
            "tensorboard": "TensorBoard summary",
            "binary_search_steps": "Binary search iterations",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.L2BrendelBethgeAttack(**params)

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
            "L2 Brendel Bethge Attack",
            self.attack_alias,
            "FB_L2BrendelBethgeAttack",
        )

    def __getstate__(self):
        del self.foolbox_attack.__dict__["_optimizer"]
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.foolbox_attack._optimizer = fb.attacks.brendel_bethge.L2Optimizer()


class LinfinityBrendelBethgeAttack(BaseAttack):
    """
    foolbox.attacks.LinfinityBrendelBethgeAttack wrapper class.

    Attack description:
    L-infinity variant of the Brendel & Bethge adversarial attack. This is a powerful
    gradient-based adversarial attack that follows the adversarial boundary (the
    boundary between the space of adversarial and non-adversarial images as defined by
    the adversarial criterion) to find the minimum distance to the clean image.

    This is the reference implementation of the Brendel & Bethge attack.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * init_attack (Optional[foolbox.attacks.base.MinimizationAttack]): (optional)
          Attack to use to find a starting points. Defaults to
          LinearSearchBlendedUniformNoiseAttack. Only used if starting_points is None.
        * overshoot (float): (optional)
        * steps (int): (optional) Maximum number of steps to run.
        * lr (float): (optional)
        * lr_decay (float): (optional)
        * lr_num_decay (int): (optional)
        * momentum (float): (optional)
        * tensorboard (Union[typing_extensions.Literal[False], None, str]): (optional)
          The log directory for TensorBoard summaries. If False, TensorBoard summaries
          will be disabled (default). If None, the logdir will be
          runs/CURRENT_DATETIME_HOSTNAME.
        * binary_search_steps (int): (optional) Number of iterations in the binary
          search. This controls the precision of the results.
        * epsilons (list): List of one or more epsilons for the attack.

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
        pars_descriptors = {
            "init_attack": "Starting attack",
            "overshoot": "overshoot",
            "steps": "Maximum steps",
            "lr": "LR",
            "lr_decay": "LR decay",
            "lr_num_decay": "LR decay number",
            "momentum": "Momentum",
            "tensorboard": "TensorBoard summary",
            "binary_search_steps": "Binary search iterations",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        foolbox_attack = fb.attacks.LinfinityBrendelBethgeAttack(**params)

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
            "L-infinity Brendel Bethge Attack",
            self.attack_alias,
            "FB_LinfinityBrendelBethgeAttack",
        )

    def __getstate__(self):
        del self.foolbox_attack.__dict__["_optimizer"]
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.foolbox_attack._optimizer = fb.attacks.brendel_bethge.LinfOptimizer()


FGM = L2FastGradientAttack
FGSM = LinfFastGradientAttack
L2PGD = L2ProjectedGradientDescentAttack
LinfPGD = LinfProjectedGradientDescentAttack
PGD = LinfPGD
