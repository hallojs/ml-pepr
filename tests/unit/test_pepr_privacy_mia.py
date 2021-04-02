import numpy as np
import pytest

from pepr.privacy.mia import Mia as mia


def test_get_target_model_indices_small():
    # Simple example (1 target model, 2 classes, shuffled)
    target_indices = np.array([10,13,7,6,11,4])
    evaluation_indices = np.array([10,18,16,11,4,13,9,1,8,6,14,7])
    record_indices_per_target = list([[3,1,0]])
    attack_test_labels = np.array([1,0,0,1,1,1,0,1,0,0,1,0])
    number_classes = 2

    res = mia._get_target_model_indices(
        target_indices,
        evaluation_indices,
        record_indices_per_target,
        attack_test_labels,
        number_classes,
    )

    assert res["indices_per_target"] == [[[9,1,2], [5,0,3]]]
    assert res["true_attack_labels_per_target"] == [
        [[True, False, False], [True, True, False]]
    ]

def test_get_target_model_indices_medium():
    # Medium example (2 target models, 2 classes, shuffled)
    target_indices = np.array([10,13,7,6,11,4])
    evaluation_indices = np.array([10,18,16,11,4,13,9,1,8,6,14,7])
    record_indices_per_target = list([[3,1,0],[2,5,4,3]])
    attack_test_labels = np.array([1,0,0,1,1,1,0,1,0,0,1,0])
    number_classes = 2

    res = mia._get_target_model_indices(
        target_indices,
        evaluation_indices,
        record_indices_per_target,
        attack_test_labels,
        number_classes,
    )

    assert res["indices_per_target"] == [
        [[9, 1, 2], [5, 0, 3]],
        [[11,9,1,2], [4,3,0,5]],
    ]
    assert res["true_attack_labels_per_target"] == [
        [[True, False, False], [True, True, False]],
        [[True, True, False, False], [True, True, False, False]],
    ]
