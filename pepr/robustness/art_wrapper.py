"""PePR wrapper classes for ART attack classes."""

import logging
import numpy as np
import os

from pepr.attack import Attack
from pepr import report
import matplotlib.pyplot as plt
from pylatex import Command, Tabular, MiniPage, NoEscape, Figure
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

    org_data = self.data[self.attack_indices_per_target[target_model_index]]
    adv_data = self.attack_results["adversarial_examples"][target_model_index]
    labels = self.labels[self.attack_indices_per_target[target_model_index]]
    is_adv = self.attack_results["is_adv"][target_model_index]
    nb_classes = np.max(labels) + 1
    nb_adv = 0

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
            is_adv[l], self.attack_results["l2_distance"][target_model_index][l]
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
                logger.debug(f"Prediction Label: {np.argmax(predicted_labels[c][idx])}")
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
    fig.savefig(save_path + f"/fig/{alias_no_spaces}-examples.pdf", bbox_inches="tight")
    plt.close(fig)


def _report_attack_results(self, save_path):
    """
    Create subsubsection describing the most important results of the attack.

    Parameters
    ----------
    self : BaseEvasionAttack or BasePatchAttack
        Base attack object.
    save_path :
        Path to save the tex, pdf and asset files of the attack report.

    This subsection contains results only for the first target model.
    """
    tm = 0  # Specify target model
    self.report_section.append(Subsubsection("Attack Results"))
    res = self.attack_results

    # Histogram
    fig = plt.figure()
    ax = plt.axes()
    ax.hist(res["success_rate_list"][tm], edgecolor="black")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of Classes")
    ax.set_axisbelow(True)

    alias_no_spaces = str.replace(self.attack_alias, " ", "_")
    fig.savefig(save_path + f"/fig/{alias_no_spaces}-hist.pdf")
    plt.close(fig)

    with self.report_section.create(MiniPage()):
        with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
            self.report_section.append(Command("centering"))
            self.report_section.append(
                Command(
                    "includegraphics",
                    NoEscape(f"fig/{alias_no_spaces}-hist.pdf"),
                    "width=8cm",
                )
            )
            self.report_section.append(Command("captionsetup", "labelformat=empty"))
            self.report_section.append(
                Command(
                    "captionof",
                    "figure",
                    extra_arguments="Success Rate Distribution",
                )
            )

        # Result table
        with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
            self.report_section.append(Command("centering"))

            with self.report_section.create(Tabular("|l|c|")) as result_tab:
                result_tab.add_hline()
                result_tab.add_row(["Success Rate", round(res["success_rate"][tm], 3)])
                result_tab.add_hline()
                result_tab.add_row(
                    ["L2 Distance", round(res["avg_l2_distance"][tm], 3)]
                )
                result_tab.add_hline()

            self.report_section.append(Command("captionsetup", "labelformat=empty"))
            self.report_section.append(
                Command("captionof", "table", extra_arguments="Attack Summary")
            )

    _plot_most_vulnerable_aes(self, save_path, tm, 10)
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
                "adversarial examples per class. They were sorted per class by the "
                "lowest distance which changes the target's prediction. Sorting per "
                "class for every n-th image may not give the absolute most "
                "vulnerable records but provides the highest diversity. "
                "(First row: Originals, second row: Adversarial examples)",
            )
        )


def _report_attack_configuration(self):
    """
    Create subsubsection about the attack and data configuration.

    Parameters
    ----------
    self : BaseEvasionAttack or BasePatchAttack
        Base attack object.
    """
    # Create tables for attack parameters and the data configuration.
    tm = 0  # Specify target model

    dc = self.data_conf
    self.report_section.append(Subsubsection("Attack Details"))
    with self.report_section.create(MiniPage()):
        with self.report_section.create(MiniPage(width=r"0.49\textwidth")):
            # -- Create table for the attack parameters.
            self.report_section.append(Command("centering"))
            with self.report_section.create(Tabular("|l|c|")) as tab_ap:
                if hasattr(self, "use_labels"):
                    tab_ap.add_hline()
                    tab_ap.add_row(["Use true labels", self.use_labels])
                self._gen_attack_pars_rows(tm, tab_ap)
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
            nr_targets, target_attack_set_size = dc["attack_indices_per_target"].shape
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


class BaseEvasionAttack(Attack):
    """
    Base ART attack class implementing the logic for running an evasion attack and
    generating a report.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    use_labels : bool
        If true, the true labels are passed to the generate function. Set true if
        `targeted` is true for a targeted attack.
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
        If true, the true labels are passed to the generate function. Set true if
        `targeted` is true for a targeted attack.
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
        * avg_l2_distance (list): Average euclidean distance (L2 norm) between original
          and perturbed images per target model.
        * success_rate_list (list): Percentage of misclassified adversarial examples
          per target model and per class.
        * l2_distance (list): Euclidean distance (L2 norm) between original
          and perturbed images for every image per target model.
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

    def art_run(self, attack_index, data, labels=None, **kwargs):
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
        kwargs :
            Additional parameters for the `generate` function of the attack.

        Returns
        -------
        An array holding the adversarial examples.
        """
        if labels is not None:
            return self.art_attacks[attack_index].generate(data, labels, **kwargs)

        return self.art_attacks[attack_index].generate(data, **kwargs)

    def run(self, **kwargs):
        """
        Run the ART attack.

        Parameters
        ----------
        kwargs :
            Additional parameters for the `generate` function of the attack.
        """
        adv_list = []
        misclass = []
        l2_dist = []
        l2_dist_avg = []
        is_adv = []

        # Run attack for every target model
        for i, art_attack in enumerate(self.art_attacks):
            logger.info(f"Attack target model ({i + 1}/{len(self.art_attacks)}).")
            data = self.data[self.attack_indices_per_target[i]]
            labels = self.labels[self.attack_indices_per_target[i]]

            if self.use_labels:
                adv = self.art_run(i, data, labels, **kwargs)
            else:
                adv = self.art_run(i, data, **kwargs)
            adv_list.append(adv)

            misclass_list = []
            l2_dist_list = []
            is_adv_list = []
            raw_diff = adv - data
            raw_diff = raw_diff.reshape(raw_diff.shape[0], -1)
            for j in range(np.max(labels) + 1):
                indices, = np.where(labels == j)

                # Find real adversarial examples and calculate accuracy on adversarial
                # examples for every class separately
                if indices.size == 0:
                    is_adv_list.append(np.NaN)
                    misclass_list.append(np.NaN)
                else:
                    p = self.target_models[i].predict(adv[indices])
                    p = np.argmax(p, axis=1)
                    p = np.not_equal(p, labels[indices])
                    is_adv_list.append(p)
                    misclass_list.append(1 - np.mean(p))

                # Calculate L2 distance of adversarial examples
                if indices.size == 0:
                    l2_dist_list.append(np.NaN)
                else:
                    l2_dist_list.append(np.linalg.norm(raw_diff[indices], axis=-1))

            l2_dist_avg_list = np.mean(np.linalg.norm(raw_diff, axis=-1))

            misclass.append(misclass_list)
            l2_dist_avg.append(l2_dist_avg_list)
            l2_dist.append(l2_dist_list)
            is_adv.append(is_adv_list)

        self.attack_results["adversarial_examples"] = adv_list
        self.attack_results["success_rate"] = np.nanmean(misclass, axis=1)
        self.attack_results["is_adv"] = is_adv
        self.attack_results["avg_l2_distance"] = l2_dist_avg
        self.attack_results["success_rate_list"] = misclass
        self.attack_results["l2_distance"] = l2_dist

        # Print every epsilon result of attack
        def _target_model_rows():
            string = ""
            for tm_i in range(len(self.target_models)):
                string = string + f"\n{f'Target Model {tm_i + 1}:':<20}"
                string = (
                    string
                    + f"{str(round(self.attack_results['success_rate'][tm_i], 3)):>10}"
                    + f"{str(round(self.attack_results['avg_l2_distance'][tm_i], 3)):>10}"
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
        _report_attack_configuration(self)
        _report_attack_results(self, save_path)

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
            if key == "targeted":
                key = "_targeted"
            elif key == "verbose":
                continue
            value = str(self.art_attacks[tm].__dict__[key])

            table.add_hline()
            table.add_row([desc, value])


class BasePatchAttack(Attack):
    """
    Base ART attack class implementing the logic for creating an adversarial patch,
    applying them to generate adversarial examples and generating a report.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
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
        * avg_l2_distance (list): Average euclidean distance (L2 norm) between original
          and perturbed images per target model.
        * success_rate_list (list): Percentage of misclassified adversarial examples
          per target model and per class.
        * l2_distance (list): Euclidean distance (L2 norm) between original
          and perturbed images for every image per target model.
    """

    def __init__(
        self,
        attack_alias,
        data,
        labels,
        attack_indices_per_target,
        target_models,
        art_attacks,
        pars_descriptors,
    ):
        super().__init__(
            attack_alias,
            {},
            data,
            labels,
            {"attack_indices_per_target": attack_indices_per_target},
            target_models,
        )

        self.attack_indices_per_target = attack_indices_per_target
        self.art_attacks = art_attacks
        self.pars_descriptors = pars_descriptors
        self.classifiers = [x.estimator for x in art_attacks]

    def art_run(self, attack_index, data, labels, kwargs_gen, kwargs_apply):
        """
        Generate patches and apply them to the given images.

        Parameters
        ----------
        attack_index : int
            Index of the corresponding target model.
        data : numpy.ndarray
            Dataset slice containing images to attack the corresponding target model.
        labels : numpy.ndarray
            Dataset slice with true labels to attack the corresponding target model.
        kwargs_gen :
            Additional parameters for the `generate` function of the attack.
        kwargs_apply :
            Additional parameters for the `apply_patch` function of the attack.

        Returns
        -------
        An array holding the adversarial examples.
        """
        if kwargs_gen is None:
            patch, p_mask = self.art_attacks[attack_index].generate(data, labels)
        else:
            patch, p_mask = self.art_attacks[attack_index].generate(
                data, labels, **kwargs_gen
            )
        self.attack_results["patch"].append(patch)
        self.attack_results["mask"].append(p_mask)

        if kwargs_apply is None:
            return self.art_attacks[attack_index].apply_patch(data)
        else:
            return self.art_attacks[attack_index].apply_patch(data, **kwargs_apply)

    def run(self, kwargs_gen=None, kwargs_apply=None):
        """
        Run the ART attack.

        Parameters
        ----------
        kwargs_gen :
            Additional parameters for the `generate` function of the attack.
        kwargs_apply :
            Additional parameters for the `apply_patch` function of the attack.
        """
        adv_list = []
        misclass = []
        l2_dist = []
        l2_dist_avg = []
        is_adv = []

        self.attack_results["patch"] = []
        self.attack_results["mask"] = []

        if kwargs_gen is not None:
            self.kwargs_gen.update(kwargs_gen)
        if kwargs_apply is not None:
            self.kwargs_apply.update(kwargs_apply)

        # Run attack for every target model
        for i, art_attack in enumerate(self.art_attacks):
            logger.info(f"Attack target model ({i + 1}/{len(self.art_attacks)}).")
            data = self.data[self.attack_indices_per_target[i]]
            labels = self.labels[self.attack_indices_per_target[i]]

            adv = self.art_run(i, data, labels, self.kwargs_gen, self.kwargs_apply)
            adv_list.append(adv)

            misclass_list = []
            l2_dist_list = []
            is_adv_list = []
            raw_diff = adv - data
            raw_diff = raw_diff.reshape(raw_diff.shape[0], -1)
            for j in range(np.max(labels) + 1):
                indices, = np.where(labels == j)

                # Find real adversarial examples and calculate accuracy on adversarial
                # examples for every class separately
                if indices.size == 0:
                    is_adv_list.append(np.NaN)
                    misclass_list.append(np.NaN)
                else:
                    p = self.target_models[i].predict(adv[indices])
                    p = np.argmax(p, axis=1)
                    p = np.not_equal(p, labels[indices])
                    is_adv_list.append(p)
                    misclass_list.append(1 - np.mean(p))

                # Calculate L2 distance of adversarial examples
                if indices.size == 0:
                    l2_dist_list.append(np.NaN)
                else:
                    l2_dist_list.append(np.linalg.norm(raw_diff[indices], axis=-1))

            l2_dist_avg_list = np.mean(np.linalg.norm(raw_diff, axis=-1))

            misclass.append(misclass_list)
            l2_dist_avg.append(l2_dist_avg_list)
            l2_dist.append(l2_dist_list)
            is_adv.append(is_adv_list)

        self.attack_results["adversarial_examples"] = adv_list
        self.attack_results["success_rate"] = np.nanmean(misclass, axis=1)
        self.attack_results["is_adv"] = is_adv
        self.attack_results["avg_l2_distance"] = l2_dist_avg
        self.attack_results["success_rate_list"] = misclass
        self.attack_results["l2_distance"] = l2_dist

        # Print every epsilon result of attack
        def _target_model_rows():
            string = ""
            for tm_i in range(len(self.target_models)):
                string = string + f"\n{f'Target Model {tm_i + 1}:':<20}"
                string = (
                    string
                    + f"{str(round(self.attack_results['success_rate'][tm_i], 3)):>10}"
                    + f"{str(round(self.attack_results['avg_l2_distance'][tm_i], 3)):>10}"
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
        _report_attack_configuration(self)
        _report_attack_results(self, save_path)

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
            if key == "targeted":
                key = "_targeted"
            elif key == "verbose":
                continue
            elif key.startswith("gen_"):
                key = key.replace("gen_", "", 1)
            elif key.startswith("apply_"):
                key = key.replace("apply_", "", 1)
            try:
                value = str(self.art_attacks[tm].__dict__[key])
            except KeyError:
                value = str(self.art_attacks[tm]._attack.__dict__[key])

            table.add_hline()
            table.add_row([desc, value])


class AdversarialPatch(BasePatchAttack):
    """
    art.attacks.evasion.AdversarialPatch wrapper class.

    Attack description:
    Implementation of the adversarial patch attack for square and rectangular images and
    videos.

    Paper link: https://arxiv.org/abs/1712.09665

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * rotation_max (float): (optional) The maximum rotation applied to random
          patches. The value is expected to be in the range [0, 180].
        * scale_min (float): (optional) The minimum scaling applied to random patches.
          The value should be in the range [0, 1], but less than scale_max.
        * scale_max (float): (optional) The maximum scaling applied to random patches.
          The value should be in the range [0, 1], but larger than scale_min.
        * learning_rate (float): (optional) The learning rate of the optimization.
        * max_iter (int): (optional) The number of optimization steps.
        * batch_size (int): (optional) The size of the training batch.
        * patch_shape: (optional) The shape of the adversarial patch as a tuple of shape
          (width, height, nb_channels). Currently only supported for
          TensorFlowV2Classifier. For classifiers of other frameworks the patch_shape is
          set to the shape of the input samples.
        * verbose (bool): (optional) Show progress bars.
        * gen_mask (numpy.ndarray): (optional) A boolean array of shape equal to the
          shape of a single samples (1, H, W) or the shape of x (N, H, W) without their
          channel dimensions. Any features for which the mask is True can be the center
          location of the patch during sampling.
        * gen_reset_patch (bool): (optional) If True reset patch to initial values of
          mean of minimal and maximal clip value, else if False (default) restart from
          previous patch values created by previous call to generate or mean of minimal
          and maximal clip value if first call to generate.
        * apply_scale (float): Scale of the applied patch in relation to the classifier
          input shape.
        * apply_patch_external: (optional) External patch to apply to the images.

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
            "rotation_max": "Max. rotation",
            "scale_min": "Min. scaling",
            "scale_max": "Max. scaling",
            "learning_rate": "Learning rate",
            "max_iter": "Max. iterations",
            "batch_size": "Batch size",
            "patch_shape": "Patch shape",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]

        self.kwargs_gen = {}
        gp = ["gen_mask", "gen_reset_patch"]
        for p in gp:
            if p in attack_pars:
                short_p = p.replace("gen_", "", 1)
                self.kwargs_gen[short_p] = attack_pars[p]

        self.kwargs_apply = {}
        ap = ["apply_scale", "apply_patch_external"]
        for p in ap:
            if p in attack_pars:
                short_p = p.replace("apply_", "", 1)
                self.kwargs_apply[short_p] = attack_pars[p]

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.AdversarialPatch(est, **params))

        super().__init__(
            attack_alias,
            data,
            labels,
            data_conf["attack_indices_per_target"],
            target_models,
            art_attacks,
            pars_descriptors,
        )

        self.report_section = report.ReportSection(
            "Adversarial Patch",
            self.attack_alias,
            "ART_AdversarialPatch",
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
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "norm": "Adversarial perturbation norm",
            "eps": "Step size",
            "eps_step": "Step size min. perturbation",
            "targeted": "Targeted attack",
            "num_random_init": "Random initialisations",
            "batch_size": "Batch size",
            "minimal": "Computing minimal perturbation",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.FastGradientMethod(est, **params))

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
            "ART_FastGradientMethod",
        )


class AutoAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.AutoAttack wrapper class.

    Attack description:
    Implementation of the AutoAttack attack.

    Paper link: https://arxiv.org/abs/2003.01690

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * norm: (optional) The norm of the adversarial perturbation. Possible values:
          “inf”, np.inf, 1 or 2.
        * eps (float): (optional) Maximum perturbation that the attacker can introduce.
        * eps_step (float): (optional) Attack step size (input variation) at each
          iteration.
        * targeted (bool): (optional) If False run only untargeted attacks, if True also
          run targeted attacks against each possible target.
        * estimator_orig (int): (optional) Original estimator to be attacked by
          adversarial examples.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * attacks (bool): (optional) The list of art.attacks.EvasionAttack attacks to be
          used for AutoAttack. If it is None or empty the standard attacks (PGD,
          APGD-ce, APGD-dlr, DeepFool, Square) will be used.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "norm": "Adversarial perturbation norm",
            "eps": "Maximum perturbation",
            "eps_step": "Step size",
            "targeted": "Targeted attack",
            "estimator_orig": "Original estimator",
            "batch_size": "Batch size",
            "attacks": "Attack objects",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.AutoAttack(est, **params))

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
            "Auto Attack",
            self.attack_alias,
            "ART_AutoAttack",
        )


class AutoProjectedGradientDescent(BaseEvasionAttack):
    """
    art.attacks.evasion.AutoProjectedGradientDescent wrapper class.

    Attack description:
    Implementation of the Auto Projected Gradient Descent attack.

    Paper link: https://arxiv.org/abs/2003.01690

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * norm: (optional) The norm of the adversarial perturbation. Possible values:
          "inf", np.inf, 1 or 2.
        * eps (float): (optional) Maximum perturbation that the attacker can introduce.
        * eps_step (float): (optional) Attack step size (input variation) at each
          iteration.
        * max_iter (int): (optional) The maximum number of iterations.
        * targeted (bool): (optional) Indicates whether the attack is targeted (True) or
          untargeted (False).
        * nb_random_init (int): (optional) Number of random initialisations within the
          epsilon ball. For num_random_init=0 starting at the original input.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * loss_type: Defines the loss to attack. Available options: None (Use loss
          defined by estimator), "cross_entropy", or "difference_logits_ratio".
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "norm": "Adversarial perturbation norm",
            "eps": "Maximum perturbation",
            "eps_step": "Step size",
            "max_iter": "Max. iterations",
            "targeted": "Targeted attack",
            "nb_random_init": "Random initialisations",
            "batch_size": "Batch size",
            "loss_type": "Loss type",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(
                art.attacks.evasion.AutoProjectedGradientDescent(est, **params)
            )

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
            "Auto Projected Gradient Descent",
            self.attack_alias,
            "ART_AutoProjectedGradientDescent",
        )


class BoundaryAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.BoundaryAttack wrapper class.

    Attack description:
    Implementation of the boundary attack from Brendel et al. (2018). This is a powerful
    black-box attack that only requires final class prediction.

    Paper link: https://arxiv.org/abs/1712.04248

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * batch_size (int): (optional) The size of the batch used by the estimator
          during inference.
        * targeted (bool): (optional) Should the attack target one specific class.
        * delta (float): (optional) Initial step size for the orthogonal step.
        * epsilon (float): (optional) Initial step size for the step towards the target.
        * step_adapt (float): (optional) Factor by which the step sizes are multiplied
          or divided, must be in the range (0, 1).
        * max_iter (int): (optional) Maximum number of iterations.
        * num_trial (int): (optional) Maximum number of trials per iteration.
        * sample_size (int): (optional) Number of samples per trial.
        * init_size (int): (optional) Maximum number of trials for initial generation of
          adversarial examples.
        * min_epsilon (float): (optional) Stop attack if perturbation is smaller than
          min_epsilon.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "batch_size": "Batch size",
            "targeted": "Targeted attack",
            "delta": "Initial step size - orthogonal",
            "epsilon": "Initial step size - target",
            "step_adapt": "Step division factor",
            "max_iter": "Max. iterations",
            "num_trial": "Max. trials",
            "sample_size": "Samples per trial",
            "init_size": "Init size",
            "min_epsilon": "Min. perturbation",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.BoundaryAttack(est, **params))

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
            "Boundary Attack",
            self.attack_alias,
            "ART_BoundaryAttack",
        )


class BrendelBethgeAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.BrendelBethgeAttack wrapper class.

    Attack description:
    Base class for the Brendel & Bethge adversarial attack, a powerful
    gradient-based adversarial attack that follows the adversarial boundary (the
    boundary between the space of adversarial and non-adversarial images as defined by
    the adversarial criterion) to find the minimum distance to the clean image.

    This is implementation of the Brendel & Bethge attack follows the reference
    implementation at
    https://github.com/bethgelab/foolbox/blob/master/foolbox/attacks/brendel_bethge.py.

    Implementation differs from the attack used in the paper in two ways:

    * The initial binary search is always using the full 10 steps (for ease of
      implementation).
    * The adaptation of the trust region over the course of optimisation is less
      greedy but is more robust, reliable and simpler (decay every K steps)

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * norm: (optional) The norm of the adversarial perturbation. Possible values:
          "inf", np.inf, 1 or 2.
        * targeted (bool): (optional) Flag determining if attack is targeted.
        * overshoot (float): (optional) If 1 the attack tries to return exactly to the
          adversarial boundary in each iteration. For higher values the attack tries to
          overshoot over the boundary to ensure that the perturbed sample in each
          iteration is  adversarial.
        * steps (int): (optional) Maximum number of iterations to run. Might converge
          and stop before that.
        * lr (float): (optional) Trust region radius, behaves similar to a learning
          rate. Smaller values decrease the step size in each iteration and ensure that
          the attack follows the boundary more faithfully.
        * lr_decay (float): (optional) The trust region lr is multiplied with lr_decay
          in regular intervals (see lr_num_decay).
        * lr_num_decay (int): (optional) Number of learning rate decays in regular
          intervals of length steps / lr_num_decay.
        * momentum (float): (optional) Averaging of the boundary estimation over
          multiple steps. A momentum of zero would always take the current estimate
          while values closer to one average over a larger number of iterations.
        * binary_search_steps (int): (optional) Number of binary search steps used to
          find the adversarial boundary between the starting point and the clean image.
        * batch_size (int): (optional) Batch size for evaluating the model for
          predictions and gradients.
        * init_size (int): (optional) Maximum number of random search steps to find
          initial adversarial example.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "norm": "Adversarial perturbation norm",
            "targeted": "Targeted attack",
            "overshoot": "Overshoot",
            "steps": "Max. iterations",
            "lr": "Trust region radius",
            "lr_decay": "lr factor",
            "lr_num_decay": "Number decays",
            "momentum": "Momentum",
            "binary_search_steps": "Binary search steps",
            "batch_size": "Batch size",
            "init_size": "Init size",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.BrendelBethgeAttack(est, **params))

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
            "Brendel Bethge Attack",
            self.attack_alias,
            "ART_BrendelBethgeAttack",
        )

    def __getstate__(self):
        for art_attack in self.art_attacks:
            del art_attack.__dict__["_optimizer"]
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        from art.attacks.evasion.brendel_bethge import (
            L0Optimizer,
            L1Optimizer,
            L2Optimizer,
            LinfOptimizer,
        )

        for art_attack in self.art_attacks:
            if art_attack.norm == 0:
                art_attack._optimizer = L0Optimizer()
            if art_attack.norm == 1:
                art_attack._optimizer = L1Optimizer()
            elif art_attack.norm == 2:
                art_attack._optimizer = L2Optimizer()
            elif art_attack.norm in ["inf", np.inf]:
                art_attack._optimizer = LinfOptimizer()


class CarliniL2Method(BaseEvasionAttack):
    """
    art.attacks.evasion.CarliniL2Method wrapper class.

    Attack description:
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is among the most
    effective and should be used among the primary attacks to evaluate potential
    defences. A major difference wrt to the original implementation
    (https://github.com/carlini/nn_robust_attacks) is that we use line search in the
    optimization of the attack objective.

    Paper link: https://arxiv.org/abs/1608.04644

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * confidence (float): (optional) Confidence of adversarial examples: a higher
          value produces examples that are farther away, from the original input, but
          classified with higher confidence as the target class.
        * targeted (bool): (optional) Should the attack target one specific class.
        * learning_rate (float): (optional) The initial learning rate for the attack
          algorithm. Smaller values produce better results but are slower to converge.
        * binary_search_steps (int): (optional) Number of times to adjust constant with
          binary search (positive value). If binary_search_steps is large, then the
          algorithm is not very sensitive to the value of initial_const. Note that the
          values gamma=0.999999 and c_upper=10e10 are hardcoded with the same values
          used by the authors of the method.
        * max_iter (int): (optional) The maximum number of iterations.
        * initial_const (float): (optional) The initial trade-off constant c to use to
          tune the relative importance of distance and confidence. If
          binary_search_steps is large, the initial constant is not important, as
          discussed in Carlini and Wagner (2016).
        * max_halving (int): (optional) Maximum number of halving steps in the line
          search optimization.
        * max_doubling (int): (optional) Maximum number of doubling steps in the line
          search optimization.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "confidence": "Confidence",
            "targeted": "Targeted attack",
            "learning_rate": "Initial learning rate",
            "binary_search_steps": "Binary search steps",
            "max_iter": "Max. iterations",
            "initial_const": "Initial trade-off constant",
            "max_halving": "Max. halving steps",
            "max_doubling": "Max. doubling steps",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.CarliniL2Method(est, **params))

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
            "Carlini L2 Method",
            self.attack_alias,
            "ART_CarliniL2Method",
        )


class CarliniLInfMethod(BaseEvasionAttack):
    """
    art.attacks.evasion.CarliniLInfMethod wrapper class.

    Attack description:
    This is a modified version of the L_2 optimized attack of Carlini and Wagner (2016).
    It controls the L_Inf norm, i.e. the maximum perturbation applied to each pixel.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * confidence (float): (optional) Confidence of adversarial examples: a higher
          value produces examples that are farther away, from the original input, but
          classified with higher confidence as the target class.
        * targeted (bool): (optional) Should the attack target one specific class.
        * learning_rate (float): (optional) The initial learning rate for the attack
          algorithm. Smaller values produce better results but are slower to converge.
        * max_iter (int): (optional) The maximum number of iterations.
        * max_halving (int): (optional) Maximum number of halving steps in the line
          search optimization.
        * max_doubling (int): (optional) Maximum number of doubling steps in the line
          search optimization.
        * eps (float): (optional) An upper bound for the L_0 norm of the adversarial
          perturbation.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "confidence": "Confidence",
            "targeted": "Targeted attack",
            "learning_rate": "Initial learning rate",
            "max_iter": "Max. iterations",
            "max_halving": "Max. halving steps",
            "max_doubling": "Max. doubling steps",
            "eps": "Upper L0 bound",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.CarliniLInfMethod(est, **params))

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
            "Carlini L-Inf Method",
            self.attack_alias,
            "ART_CarliniLInfMethod",
        )


class DeepFool(BaseEvasionAttack):
    """
    art.attacks.evasion.DeepFool wrapper class.

    Attack description:
    Implementation of the attack from Moosavi-Dezfooli et al. (2015).

    Paper link: https://arxiv.org/abs/1511.04599

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * max_iter (int): (optional) The maximum number of iterations.
        * epsilon (float): (optional) Overshoot parameter.
        * nb_grads (int): (optional) The number of class gradients (top nb_grads w.r.t.
          prediction) to compute. This way only the most likely classes are considered,
          speeding up the computation.
        * batch_size (int): (optional) Batch size
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "max_iter": "Max. iterations",
            "epsilon": "Overshoot parameter",
            "nb_grads": "Class gradients",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.DeepFool(est, **params))

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
            "Deep Fool",
            self.attack_alias,
            "ART_DeepFool",
        )


class ElasticNet(BaseEvasionAttack):
    """
    art.attacks.evasion.ElasticNet wrapper class.

    Attack description:
    The elastic net attack of Pin-Yu Chen et al. (2018).

    Paper link: https://arxiv.org/abs/1709.04114

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * confidence (float): (optional) Confidence of adversarial examples: a higher
          value produces examples that are farther away, from the original input, but
          classified with higher confidence as the target class.
        * targeted (bool): (optional) Should the attack target one specific class.
        * learning_rate (float): (optional) The initial learning rate for the attack
          algorithm. Smaller values produce better results but are slower to converge.
        * binary_search_steps (int): (optional) Number of times to adjust constant with
          binary search (positive value).
        * max_iter (int): (optional) The maximum number of iterations.
        * beta (float): (optional) Hyperparameter trading off L2 minimization for L1
          minimization.
        * initial_const (float): (optional) The initial trade-off constant c to use to
          une the relative importance of distance and confidence. If binary_search_steps
          is large, the initial constant is not important, as discussed in Carlini and
          Wagner (2016).
        * batch_size (int): (optional) Internal size of batches on which adversarial
          samples are generated.
        * decision_rule (str): (optional) Decision rule. ‘EN’ means Elastic Net rule,
          'L1' means L1 rule, 'L2' means L2 rule.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "confidence": "Confidence",
            "targeted": "Targeted attack",
            "learning_rate": "Initial learning rate",
            "binary_search_steps": "Binary search steps",
            "max_iter": "Max. iterations",
            "beta": "L2 for L1 trade-off",
            "initial_const": "Initial trade-off constant",
            "batch_size": "Batch size",
            "decision_rule": "Decision rule",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.ElasticNet(est, **params))

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
            "Elastic Net",
            self.attack_alias,
            "ART_ElasticNet",
        )


class FeatureAdversaries(BaseEvasionAttack):
    """
    art.attacks.evasion.FeatureAdversaries wrapper class.

    Attack description:
    This class represent a Feature Adversaries evasion attack.

    Paper link: https://arxiv.org/abs/1511.05122

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * delta: (optional) The maximum deviation between source and guide images.
        * layer: (optional) Index of the representation layer.
        * batch_size (int): (optional) Batch size.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "delta": "Max. deviation",
            "layer": "Representation layer",
            "batch_size": "Batch size",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.FeatureAdversaries(est, **params))

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
            "Feature Adversaries",
            self.attack_alias,
            "ART_FeatureAdversaries",
        )


class FrameSaliencyAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.FrameSaliencyAttack wrapper class.

    Attack description:
    Implementation of the attack framework proposed by Inkawhich et al. (2018).
    Prioritizes the frame of a sequential input to be adversarially perturbed based on
    the saliency score of each frame.

    Paper link: https://arxiv.org/abs/1811.11875

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * attacker (EvasionAttack): (optional) An adversarial evasion attacker which
          supports masking. Currently supported: ProjectedGradientDescent,
          BasicIterativeMethod, FastGradientMethod.
        * method (str): (optional) Specifies which method to use: "iterative_saliency"
          (adds perturbation iteratively to frame with highest saliency score until
          attack is successful), "iterative_saliency_refresh" (updates perturbation
          after each iteration), "one_shot" (adds all perturbations at once, i.e.
          defaults to original attack).
        * frame_index (int): (optional) Index of the axis in input (feature) array x
          representing the frame dimension.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * verbose (bool): (optional) Show progress bars.

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
            "attacker": "Evasion attacker",
            "method": "Method",
            "frame_index": "Frame index",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        use_labels = True  # Labels required

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.FrameSaliencyAttack(est, **params))

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
            "Frame Saliency Attack",
            self.attack_alias,
            "ART_FrameSaliencyAttack",
        )


class HopSkipJump(BaseEvasionAttack):
    """
    art.attacks.evasion.HopSkipJump wrapper class.

    Attack description:
    Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a
    powerful black-box attack that only requires final class prediction, and is an
    advanced version of the boundary attack.

    Paper link: https://arxiv.org/abs/1904.02144

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * batch_size (int): (optional) The size of the batch used by the estimator
          during inference.
        * targeted (bool): (optional) Should the attack target one specific class.
        * norm:(optional) Order of the norm. Possible values: "inf", np.inf or 2.
        * max_iter (int): (optional) Maximum number of iterations.
        * max_eval (int): (optional) Maximum number of evaluations for estimating
          gradient.
        * init_eval (int): (optional) Initial number of evaluations for estimating
          gradient.
        * init_size (int): (optional) Maximum number of trials for initial generation of
          adversarial examples.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "batch_size": "Batch size",
            "targeted": "Targeted attack",
            "norm": "Adversarial perturbation norm",
            "max_iter": "Max. iterations",
            "max_eval": "Max. evaluations",
            "init_eval": "Initial evaluations",
            "init_size": "Max. trials",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.HopSkipJump(est, **params))

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
            "HopSkipJump",
            self.attack_alias,
            "ART_HopSkipJump",
        )


class BasicIterativeMethod(BaseEvasionAttack):
    """
    art.attacks.evasion.BasicIterativeMethod wrapper class.

    Attack description:
    The Basic Iterative Method is the iterative version of FGM and FGSM.

    Paper link: https://arxiv.org/abs/1607.02533

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * eps: (optional) Maximum perturbation that the attacker can introduce.
        * eps_step: (optional) Attack step size (input variation) at each iteration.
        * max_iter (int): (optional) The maximum number of iterations.
        * targeted (bool): (optional) Indicates whether the attack is targeted (True) or
          untargeted (False).
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "eps": "Max. perturbation",
            "eps_step": "Step size",
            "max_iter": "Max. iterations",
            "targeted": "Targeted attack",
            "batch_size": "Batch size",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.BasicIterativeMethod(est, **params))

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
            "Basic Iterative Method",
            self.attack_alias,
            "ART_BasicIterativeMethod",
        )


class ProjectedGradientDescent(BaseEvasionAttack):
    """
    art.attacks.evasion.ProjectedGradientDescent wrapper class.

    Attack description:
    The Projected Gradient Descent attack is an iterative method in which, after each
    iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the
    permitted data range). This is the attack proposed by Madry et al. for adversarial
    training.

    Paper link: https://arxiv.org/abs/1706.06083

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * norm: (optional) The norm of the adversarial perturbation supporting "inf",
          np.inf, 1 or 2.
        * eps: (optional) Maximum perturbation that the attacker can introduce.
        * eps_step: (optional) Attack step size (input variation) at each iteration.
        * random_eps (bool): (optional) When True, epsilon is drawn randomly from
          truncated normal distribution. The literature suggests this for FGSM based
          training to generalize across different epsilons. eps_step is modified to
          preserve the ratio of eps / eps_step. The effectiveness of this method with
          PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).
        * max_iter (int): (optional) The maximum number of iterations.
        * targeted (bool): (optional) Indicates whether the attack is targeted (True) or
          untargeted (False).
        * num_random_init (int): (optional) Number of random initialisations within the
          epsilon ball. For num_random_init=0 starting at the original input.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "norm": "Adversarial perturbation norm",
            "eps": "Max. perturbation",
            "eps_step": "Step size",
            "random_eps": "Random epsilons",
            "max_iter": "Max. iterations",
            "targeted": "Targeted attack",
            "num_random_init": "Random initialisations",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(
                art.attacks.evasion.ProjectedGradientDescent(est, **params)
            )

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
            "Projected Gradient Descent",
            self.attack_alias,
            "ART_ProjectedGradientDescent",
        )


class NewtonFool(BaseEvasionAttack):
    """
    art.attacks.evasion.NewtonFool wrapper class.

    Attack description:
    Implementation of the attack from Uyeong Jang et al. (2017).

    Paper link: http://doi.acm.org/10.1145/3134600.3134635

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * max_iter (int): (optional) The maximum number of iterations.
        * eta (float): (optional) The eta coefficient.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * verbose (bool): (optional) Show progress bars.

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
            "max_iter": "Max. iterations",
            "eta": "eta coefficient",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        use_labels = True  # Labels required

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.NewtonFool(est, **params))

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
            "Newton Fool",
            self.attack_alias,
            "ART_NewtonFool",
        )


class PixelAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.PixelAttack wrapper class.

    Attack description:
    This attack was originally implemented by Vargas et al. (2019). It is generalisation
    of One Pixel Attack originally implemented by Su et al. (2019).

    One Pixel Attack Paper link:
    https://ieeexplore.ieee.org/abstract/document/8601309/citations#citations
    (arXiv link: https://arxiv.org/pdf/1710.08864.pdf)
    Pixel Attack Paper link: https://arxiv.org/abs/1906.06026

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * th: (optional) threshold value of the Pixel/ Threshold attack. th=None
          indicates finding a minimum threshold.
        * es (int): (optional) Indicates whether the attack uses CMAES (0) or DE (1) as
          Evolutionary Strategy.
        * targeted (bool): (optional) Indicates whether the attack is targeted (True) or
          untargeted (False).
        * verbose (bool): (optional) Indicates whether to print verbose messages of ES
          used.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "th": "Pixel threshold",
            "es": "Evolutionary strategy",
            "targeted": "Targeted attack",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.PixelAttack(est, **params))

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
            "Pixel Attack",
            self.attack_alias,
            "ART_PixelAttack",
        )


class ThresholdAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.ThresholdAttack wrapper class.

    Attack description:
    This attack was originally implemented by Vargas et al. (2019).

    Paper link: https://arxiv.org/abs/1906.06026

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * th: (optional) threshold value of the Pixel/ Threshold attack. th=None
          indicates finding a minimum threshold.
        * es (int): (optional) Indicates whether the attack uses CMAES (0) or DE (1) as
          Evolutionary Strategy.
        * targeted (bool): (optional) Indicates whether the attack is targeted (True) or
          untargeted (False).
        * verbose (bool): (optional) Indicates whether to print verbose messages of ES
          used.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "th": "Pixel threshold",
            "es": "Evolutionary strategy",
            "targeted": "Targeted attack",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.ThresholdAttack(est, **params))

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
            "Threshold Attack",
            self.attack_alias,
            "ART_ThresholdAttack",
        )


class SaliencyMapMethod(BaseEvasionAttack):
    """
    art.attacks.evasion.SaliencyMapMethod wrapper class.

    Attack description:
    Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).

    Paper link: https://arxiv.org/abs/1511.07528

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * theta (float): (optional) Amount of Perturbation introduced to each modified
          feature per step (can be positive or negative).
        * gamma (float): (optional) Maximum fraction of features being perturbed
          (between 0 and 1).
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "theta": "Perturbation amount",
            "gamma": "Max. fraction",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.SaliencyMapMethod(est, **params))

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
            "Saliency Map Method",
            self.attack_alias,
            "ART_SaliencyMapMethod",
        )


class SimBA(BaseEvasionAttack):
    """
    art.attacks.evasion.SimBA wrapper class.

    Attack description:
    This class implements the black-box attack SimBA.

    Paper link: https://arxiv.org/abs/1905.07121

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * attack (str): (optional) attack type: pixel (px) or DCT (dct) attacks
        * max_iter (int): (optional) The maximum number of iterations.
        * epsilon (float): (optional) Overshoot parameter.
        * order (str): (optional) order of pixel attacks: random or diagonal (diag)
        * freq_dim (int): (optional) dimensionality of 2D frequency space (DCT).
        * stride (int): (optional) stride for block order (DCT).
        * targeted (bool): (optional) perform targeted attack
        * batch_size (int): (optional) Batch size (but, batch process unavailable in
          this implementation)

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
            "attack": "Attack type",
            "max_iter": "Max. iterations",
            "epsilon": "Overshoot",
            "order": "Order",
            "freq_dim": "DCT dimensionality",
            "stride": "Block order stride",
            "targeted": "Targeted attack",
            "batch_size": "Batch size",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        use_labels = True  # Labels required

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.SimBA(est, **params))

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
            "SimBA",
            self.attack_alias,
            "ART_SimBA",
        )


class SpatialTransformation(BaseEvasionAttack):
    """
    art.attacks.evasion.SpatialTransformation wrapper class.

    Attack description:
    Implementation of the spatial transformation attack using translation and rotation
    of inputs. The attack conducts black-box queries to the target model in a grid
    search over possible translations and rotations to find optimal attack parameters.

    Paper link: https://arxiv.org/abs/1712.02779

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * max_translation (float): (optional) The maximum translation in any direction
          as percentage of image size. The value is expected to be in the range
          [0, 100].
        * num_translations (int): (optional) The number of translations to search on
          grid spacing per direction.
        * max_rotation (float): (optional) The maximum rotation in either direction in
          degrees. The value is expected to be in the range [0, 180].
        * num_rotations (int): (optional) The number of rotations to search on grid
          spacing.
        * verbose (bool): (optional) Show progress bars.

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
            "max_translation": "Max. translation",
            "num_translations": "Translations",
            "max_rotation": "Max. rotation",
            "num_rotations": "Rotations",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        use_labels = True  # Labels required

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.SpatialTransformation(est, **params))

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
            "Spatial Transformation",
            self.attack_alias,
            "ART_SpatialTransformation",
        )


class SquareAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.SquareAttack wrapper class.

    Attack description:
    This class implements the SquareAttack attack.

    Paper link: https://arxiv.org/abs/1912.00049

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * norm: (optional) The norm of the adversarial perturbation. Possible values:
          "inf", np.inf, 1 or 2.
        * max_iter (int): (optional) Maximum number of iterations.
        * eps (float): (optional) Maximum perturbation that the attacker can introduce.
        * p_init (float): (optional) Initial fraction of elements.
        * nb_restarts (int): (optional) Number of restarts.
        * batch_size (int): (optional) Batch size for estimator evaluations.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "norm": "Adversarial perturbation norm",
            "max_iter": "Max. iterations",
            "eps": "Max. perturbation",
            "p_init": "Initial fraction",
            "nb_restarts": "Restarts",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.SquareAttack(est, **params))

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
            "Square Attack",
            self.attack_alias,
            "ART_SquareAttack",
        )


class TargetedUniversalPerturbation(BaseEvasionAttack):
    """
    art.attacks.evasion.TargetedUniversalPerturbation wrapper class.

    Attack description:
    Implementation of the attack from Hirano and Takemoto (2019). Computes a fixed
    perturbation to be applied to all future inputs. To this end, it can use any
    adversarial attack method.

    Paper link: https://arxiv.org/abs/1911.06502

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * attacker (str): (optional) Adversarial attack name. Default is 'deepfool'.
          Supported names: 'fgsm'.
        * attacker_params: (optional) Parameters specific to the adversarial attack. If
          this parameter is not specified, the default parameters of the chosen attack
          will be used.
        * delta (float): (optional) desired accuracy
        * max_iter (int): (optional) The maximum number of iterations for computing
          universal perturbation.
        * eps (float): (optional) Attack step size (input variation)
        * norm: (optional) The norm of the adversarial perturbation. Possible values:
          "inf", np.inf, 2

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
            "attacker": "Attacker",
            "attacker_params": "Attacker parameters",
            "delta": "Desired accuracy",
            "max_iter": "Max. iterations",
            "eps": "Step size",
            "norm": "Adversarial perturbation norm",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        use_labels = True  # Labels required

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(
                art.attacks.evasion.TargetedUniversalPerturbation(est, **params)
            )

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
            "Targeted Universal Perturbation",
            self.attack_alias,
            "ART_TargetedUniversalPerturbation",
        )


class UniversalPerturbation(BaseEvasionAttack):
    """
    art.attacks.evasion.UniversalPerturbation wrapper class.

    Attack description:
    Implementation of the attack from Moosavi-Dezfooli et al. (2016). Computes a fixed
    perturbation to be applied to all future inputs. To this end, it can use any
    adversarial attack method.

    Paper link: https://arxiv.org/abs/1610.08401

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * attacker (str): (optional) Adversarial attack name. Adversarial attack name.
          Default is 'deepfool'. Supported names: 'carlini', 'carlini_inf', 'deepfool',
          'fgsm', 'bim', 'pgd', 'margin', 'ead', 'newtonfool', 'jsma', 'vat', 'simba'.
        * attacker_params: (optional) Parameters specific to the adversarial attack. If
          this parameter is not specified, the default parameters of the chosen attack
          will be used.
        * delta (float): (optional) desired accuracy
        * max_iter (int): (optional) The maximum number of iterations for computing
          universal perturbation.
        * eps (float): (optional) Attack step size (input variation)
        * norm: (optional) The norm of the adversarial perturbation. Possible values:
          "inf", np.inf, 2
        * batch_size (int): (optional) Batch size for model evaluations in
          UniversalPerturbation.
        * verbose (bool): (optional) Show progress bars.

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
            "attacker": "Attacker",
            "attacker_params": "Attacker parameters",
            "delta": "Desired accuracy",
            "max_iter": "Max. iterations",
            "eps": "Step size",
            "norm": "Adversarial perturbation norm",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        use_labels = True  # Labels required

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.UniversalPerturbation(est, **params))

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
            "Universal Perturbation",
            self.attack_alias,
            "ART_UniversalPerturbation",
        )


class VirtualAdversarialMethod(BaseEvasionAttack):
    """
    art.attacks.evasion.VirtualAdversarialMethod wrapper class.

    Attack description:
    This attack was originally proposed by Miyato et al. (2016) and was used for virtual
    adversarial training.

    Paper link: https://arxiv.org/abs/1507.00677

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * eps (float): (optional) Attack step (max input variation).
        * finite_diff (float): (optional) The finite difference parameter.
        * max_iter (int): (optional) The maximum number of iterations.
        * batch_size (int): (optional) Size of the batch on which adversarial samples
          are generated.
        * verbose (bool): (optional) Show progress bars.

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
            "eps": "Attack step",
            "finite_diff": "Difference parameter",
            "max_iter": "Max. iterations",
            "batch_size": "Batch size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        use_labels = True  # Labels required

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(
                art.attacks.evasion.VirtualAdversarialMethod(est, **params)
            )

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
            "Virtual Adversarial Method",
            self.attack_alias,
            "ART_VirtualAdversarialMethod",
        )


class ZooAttack(BaseEvasionAttack):
    """
    art.attacks.evasion.ZooAttack wrapper class.

    Attack description:
    The black-box zeroth-order optimization attack from Pin-Yu Chen et al. (2018). This
    attack is a variant of the C&W attack which uses ADAM coordinate descent to perform
    numerical estimation of gradients.

    Paper link: https://arxiv.org/abs/1708.03999

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    attack_pars : dict
        Dictionary containing all needed attack parameters:

        * confidence (float): (optional) Confidence of adversarial examples: a higher
          value produces examples that are farther away, from the original input, but
          classified with higher confidence as the target class.
        * targeted (bool): (optional) Should the attack target one specific class.
        * learning_rate (float): (optional) The initial learning rate for the attack
          algorithm. Smaller values produce better results but are slower to converge.
        * max_iter (int): (optional) The maximum number of iterations.
        * binary_search_steps (int): (optional) Number of times to adjust constant with
          binary search (positive value).
        * initial_const (float): (optional) The initial trade-off constant c to use to
          tune the relative importance of distance and confidence. If
          binary_search_steps is large, the initial constant is not important, as
          discussed in Carlini and Wagner (2016).
        * abort_early (bool): (optional) True if gradient descent should be abandoned
          when it gets stuck.
        * use_resize (bool): (optional) True if to use the resizing strategy from the
          paper: first, compute attack on inputs resized to 32x32, then increase size if
          needed to 64x64, followed by 128x128.
        * use_importance (bool): (optional) True if to use importance sampling when
          choosing coordinates to update.
        * nb_parallel (int): (optional) Number of coordinate updates to run in parallel.
          A higher value for nb_parallel should be preferred over a large batch size.
        * batch_size (int): (optional) Internal size of batches on which adversarial
          samples are generated. Small batch sizes are encouraged for ZOO, as the
          algorithm already runs nb_parallel coordinate updates in parallel for each
          sample. The batch size is a multiplier of nb_parallel in terms of memory
          consumption.
        * variable_h (float): (optional) Step size for numerical estimation of
          derivatives.
        * verbose (bool): (optional) Show progress bars.
        * use_labels (bool): (optional) If true, the true labels are passed to the
          attack as target labels.

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
            "confidence": "Confidence",
            "targeted": "Targeted attack",
            "learning_rate": "Learning rate",
            "max_iter": "Max. iterations",
            "binary_search_steps": "Binary search steps",
            "initial_const": "Initial cost",
            "abort_early": "Abort early",
            "use_resize": "Resizing strategy",
            "use_importance": "Used importance",
            "nb_parallel": "Parallel updates",
            "batch_size": "Batch size",
            "variable_h": "Step size",
            "verbose": "Verbose output",
        }

        # Handle specific attack class parameters
        params = {}
        for k in pars_descriptors:
            if k in attack_pars:
                params[k] = attack_pars[k]
        if "use_labels" in attack_pars:
            use_labels = attack_pars["use_labels"]
        else:
            use_labels = False

        art_attacks = []
        for target_model in target_models:
            est = KerasClassifier(target_model, clip_values=(0, 1))
            art_attacks.append(art.attacks.evasion.ZooAttack(est, **params))

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
            "Zoo Attack",
            self.attack_alias,
            "ART_ZooAttack",
        )
