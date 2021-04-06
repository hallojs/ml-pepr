import numpy as np
import pytest

from pepr.privacy.mia import Mia as mia


def test_get_target_model_indices_small():
    # Simple example (1 target model, 2 classes, shuffled)
    target_indices = np.array([10, 13, 7, 6, 11, 4])
    evaluation_indices = np.array([10, 18, 16, 11, 4, 13, 9, 1, 8, 6, 14, 7])
    record_indices_per_target = np.array([np.array([3, 1, 0])])
    attack_test_labels = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
    number_classes = 2

    res = mia._get_target_model_indices(
        target_indices,
        evaluation_indices,
        record_indices_per_target,
        attack_test_labels,
        number_classes,
    )

    assert res["indices_per_target"] == [[[9, 1, 2], [5, 0, 3]]]
    assert res["true_attack_labels_per_target"] == [
        [[True, False, False], [True, True, False]]
    ]


def test_get_target_model_indices_medium():
    # Medium example (2 target models, 2 classes, shuffled)
    target_indices = np.array([10, 13, 7, 6, 11, 4])
    evaluation_indices = np.array([10, 18, 16, 11, 4, 13, 9, 1, 8, 6, 14, 7])
    record_indices_per_target = np.array([np.array([3, 1, 0]), np.array([2, 5, 4, 3])])
    attack_test_labels = np.array([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
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
        [[11, 9, 1, 2], [4, 3, 0, 5]],
    ]
    assert res["true_attack_labels_per_target"] == [
        [[True, False, False], [True, True, False]],
        [[True, True, False, False], [True, True, False, False]],
    ]


def test_attack_model_evaluation():
    class Mock_tf_model:
        def predict(self, prediction_out):
            return prediction_out

    # Test with 2 target models and 3 classes
    attack_models = [
        Mock_tf_model(),
        Mock_tf_model(),
        Mock_tf_model(),
    ]
    attack_test_data = [
        [
            np.array([[0], [0], [1]]),
            np.array([[0], [1], [0], [1]]),
            np.array([[1], [1], [0]]),
        ],
        [
            np.array([[0], [1], [0], [0]]),
            np.array([[1], [0], [1]]),
            np.array([[1], [1], [0]]),
        ],
    ]
    target_data = {
        "indices_per_target": [
            [[6, 0, 3], [19, 4, 1, 7], [5, 2, 8]],
            [[18, 3, 0, 6], [7, 1, 4], [2, 5, 8]],
        ],
        "true_attack_labels_per_target": [
            [[True, False, False], [True, True, True, False], [True, False, False]],
            [[True, True, True, False], [True, False, False], [True, False, False]],
        ],
    }

    res = mia._attack_model_evaluation(attack_models, attack_test_data, target_data)

    assert np.allclose(res["tn_list"], [[1, 1], [0, 1], [1, 1]])
    assert np.allclose(res["tp_list"], [[0, 1], [1, 1], [1, 1]])
    assert np.allclose(res["fn_list"], [[1, 2], [2, 0], [0, 0]])
    assert np.allclose(res["fp_list"], [[1, 0], [1, 1], [1, 1]])
    assert np.allclose(
        res["eval_accuracy_list"], [[1 / 3, 1 / 2], [1 / 4, 2 / 3], [2 / 3, 2 / 3]]
    )
    assert np.allclose(res["precision_list"], [[0, 1], [1 / 2, 1 / 2], [1 / 2, 1 / 2]])
    assert np.allclose(res["recall_list"], [[0, 1 / 3], [1 / 3, 1], [1, 1]])
    assert np.allclose(res["eval_accuracy"], [5 / 12, 11 / 18])
    assert np.allclose(res["precision"], [1 / 3, 2 / 3])
    assert np.allclose(res["recall"], [4 / 9, 7 / 9])
    assert np.allclose(res["overall_eval_accuracy"], 37 / 72)
    assert np.allclose(res["overall_precision"], 1 / 2)
    assert np.allclose(res["overall_recall"], 11 / 18)
