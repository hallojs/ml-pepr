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
