import numpy as np
import pytest

from pepr.privacy.mia import Mia as mia


def test_get_target_model_indices():
    # Simple example (1 target model, 2 classes, sorted)
    eval_set_size = 10
    target_indices = np.array([0, 1, 4, 6, 9])
    record_indices_per_target = list([[1, 3, 4]])
    attack_test_labels = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    number_classes = 2

    res = mia._get_target_model_indices(
        eval_set_size,
        target_indices,
        record_indices_per_target,
        attack_test_labels,
        number_classes,
    )

    assert res["indices_per_target"] == [[[1, 6, 9], [0, 2, 3]]]
    assert res["true_attack_labels_per_target"] == [
        [[True, True, True], [False, False, False]]
    ]
    # Bigger example (2 target models, 3 classes, shuffled)
    eval_set_size = 20
    target_indices = np.array([0, 5, 4, 2, 7, 1, 3, 6, 18, 19])
    record_indices_per_target = list([[9, 1, 7, 2, 5], [4, 3, 8, 6, 0]])
    attack_test_labels = np.array(
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
    )
    number_classes = 3

    res = mia._get_target_model_indices(
        eval_set_size,
        target_indices,
        record_indices_per_target,
        attack_test_labels,
        number_classes,
    )

    assert res["indices_per_target"] == [
        [[6, 0, 3], [19, 4, 1, 7], [5, 2, 8]],
        [[18, 3, 0, 6], [7, 1, 4], [2, 5, 8]],
    ]
    assert res["true_attack_labels_per_target"] == [
        [[True, False, False], [True, True, True, False], [True, False, False]],
        [[True, True, True, False], [True, False, False], [True, False, False]],
    ]
