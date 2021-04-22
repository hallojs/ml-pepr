import numpy as np
import pytest

from pepr.privacy.gmia import DirectGmia as dgmia


def test_attack_evaluation():
    # target records: 0, 1, 2, 3, 4
    infered_members_p = [
        np.array([0, 1, 2]),
        np.array([0, 3]),
        np.array([1, 2, 4]),
        np.array([1, 2, 3]),
    ]

    # Simplified assumption: The infered members and non-members are for all 101
    # cut-off-p-values the same.
    infered_members = 101 * [infered_members_p]

    infered_non_members_p = [
        np.array([3, 4]),
        np.array([1, 2, 4]),
        np.array([0, 3]),
        np.array([0, 4]),
    ]

    infered_non_members = 101 * [infered_non_members_p]

    records_per_target_model = np.array(
        [[2, 8, 4, 9, 1], [6, 7, 3, 0, 5], [2, 9, 6, 4, 0], [3, 1, 7, 8, 5]]
    )

    attack_results = dgmia._attack_evaluation(
        infered_members, infered_non_members, records_per_target_model
    )

    assert attack_results["tp_list"] == 101 * [[2, 2, 2, 2]]
    assert attack_results["fp_list"] == 101 * [[1, 0, 1, 1]]
    assert attack_results["fn_list"] == 101 * [[1, 0, 1, 0]]
    assert attack_results["tn_list"] == 101 * [[1, 3, 1, 2]]
    assert attack_results["precision_list"] == 101 * [[2/3, 1, 2/3, 2/3]]
    assert attack_results["recall_list"] == 101 * [[2/3, 1, 2/3, 1]]

    assert attack_results["overall_precision"] == pytest.approx(101 * [3/4])
