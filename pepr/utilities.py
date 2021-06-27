"""Utility module containing utilities to speed up pentesting."""

import numpy as np


def assign_record_ids_to_target_models(
    target_knowledge_size, number_target_models, target_training_set_size, offset=0
):
    """Create training datasets (index sets) for the target models.

    Each target training dataset contains exactly 50 percent of the given data. Each
    record is in the half of the data sets.

    Parameters
    ----------
    target_knowledge_size : int
        Number of data samples to be distributed to the target data sets.
    number_target_models : int
        Number of target models for which datasets should be created.
    target_training_set_size : int
        Number of records each target training set should be contain.
    offset : int
        If offset is zero, the lowest index in the resulting datasets is zero. If the
        offset is o, all indices i are shifted by o: i + o.
    Returns
    -------
    numpy.ndarray
        Index array that describes which record is used to train which model.
    """
    records_per_target_model = np.array([])
    for i in range(0, int(number_target_models / 2)):
        np.random.seed(i)
        selection = np.random.choice(
            np.arange(target_knowledge_size),
            target_knowledge_size,
            replace=False,
        )
        if i > 0:
            records_per_target_model = np.vstack(
                (records_per_target_model, selection[:target_training_set_size])
            )
            records_per_target_model = np.vstack(
                (records_per_target_model, selection[target_training_set_size:])
            )
        else:
            records_per_target_model = np.vstack(
                (
                    selection[:target_training_set_size],
                    selection[target_training_set_size:],
                )
            )

    return records_per_target_model + offset


def plot_class_dist_histogram(attack_alias, class_data, save_path):
    """
    Plot a class distribution histogram.

    Parameters
    ----------
    attack_alias : str
        Alias for a specific instantiation of the class.
    class_data : list
        List of data to plot per class.
    save_path : str
        Path to save the plotted figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes()
    ax.hist(class_data, edgecolor="black")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Number of Classes")
    ax.set_axisbelow(True)
    alias_no_spaces = str.replace(attack_alias, " ", "_")
    path = f"fig/{alias_no_spaces}-hist.pdf"
    fig.savefig(save_path + f"/{path}")
    plt.close(fig)
    return path


def create_attack_pars_table(report_section, values, pars_descriptors):
    """
    Generate LaTex table from pars_descriptors with fancy parameter descriptions.

    Parameters
    ----------
    report_section : pylatex.section.Section
        Section in LaTex report for the current attack.
    values : dict
        Dictionary with the values stored for the keys in `pars_descriptors`. If
        `pars_descriptors` has an entry "max_iter", then `values` should also have an
        entry "max_iter" but holding the argument value instead of a description.
    pars_descriptors : dict
        Dictionary of attack parameters and their description shown in the attack
        report.

    Returns
    -------
    int
        Number of table rows added.
    """
    from pylatex import Command, Tabular

    new_rows = len(pars_descriptors)

    if new_rows > 0:
        # Create table for the attack parameters.
        new_rows = len(pars_descriptors)
        with report_section.create(Tabular("|l|c|")) as tab_ap:
            for key in pars_descriptors:
                # Add table row per pars_descriptor entry
                desc = pars_descriptors[key]
                if isinstance(values[key], float):
                    value = str(round(values[key], 3))
                else:
                    value = str(values[key])
                tab_ap.add_hline()
                tab_ap.add_row([desc, value])
            tab_ap.add_hline()
        report_section.append(Command("captionsetup", "labelformat=empty"))
        report_section.append(
            Command(
                "captionof",
                "table",
                extra_arguments="Attack parameters",
            )
        )
    else:
        report_section.append("Attack has no configuration parameters.")

    return new_rows
