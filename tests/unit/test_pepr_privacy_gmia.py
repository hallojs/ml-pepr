import numpy as np
from pepr.privacy.gmia import DirectGmia as dgmia


def test_attack_evaluation():
    # target records: 0, 1, 2, 3, 4
    infered_members = [
        np.array([0, 1, 2]),
        np.array([0, 3]),
        np.array([1, 2, 4]),
        np.array([1, 2, 3]),
    ]

    infered_non_members = [
        np.array([3, 4]),
        np.array([1, 2, 4]),
        np.array([0, 3]),
        np.array([0, 4]),
    ]

    records_per_target_model = np.array(
        [[2, 8, 4, 9, 1], [6, 7, 3, 0, 5], [2, 9, 6, 4, 0], [3, 1, 7, 8, 5]]
    )

    attack_results = dgmia._attack_evaluation(
        infered_members, infered_non_members, records_per_target_model
    )

    assert attack_results["tp_list"] == [2, 2, 2, 2]
    assert attack_results["fp_list"] == [1, 0, 1, 1]
    assert attack_results["fn_list"] == [1, 0, 1, 0]
    assert attack_results["tn_list"] == [1, 3, 1, 2]
    assert attack_results["accuracy_list"] == [2/3, 1, 2/3, 2/3]

    assert attack_results["overall_accuracy"] == 8/11
