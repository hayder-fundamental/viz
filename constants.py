import enum

# Can use these to reproduce the offline eval score weightings.
offline_eval_num_context_probs = [1.0, 0.75818182, 0.24, 0.13636364]


class Tags:
    classification = "classification"
    regression = "regression"
    relevant = "Hayder::relevant"

    # TODO(HE): move this w/ refactor
    ms4_sota = "Hayder::MS4-SOTA"


class RunStatus(enum.StrEnum):
    RUNNING = "running"
    FINISHED = "finished"
    # , , crashed, killed, preempting, preempted


class MetricNames:
    class OnlineEval:
        _eip_stub = "evaluation_Improvement probability over TabPFNv2/score"
        _acc = "at  - accuracy"
        _mse = "at  - mse"

        eip_acc = f"{_eip_stub} {_acc}"
        eip_acc_32 = f"{_eip_stub} {_acc} with 32 num_context"
        eip_acc_128 = f"{_eip_stub} {_acc} with 128 num_context"
        eip_acc_1024 = f"{_eip_stub} {_acc} with 1024 num_context"
        eip_acc_7500 = f"{_eip_stub} {_acc} with 7500 num_context"
        eip_acc_10000 = f"{_eip_stub} {_acc} with 10000 num_context"
        eip_mse = f"{_eip_stub} {_mse}"
        eip_mse_32 = f"{_eip_stub} {_mse} with 32 num_context"
        eip_mse_128 = f"{_eip_stub} {_mse} with 128 num_context"
        eip_mse_1024 = f"{_eip_stub} {_mse} with 1024 num_context"
        eip_mse_7500 = f"{_eip_stub} {_mse} with 7500 num_context"
        eip_score = "evaluation_Improvement probability over TabPFNv2/main score"
        eip_score_128 = f"{_eip_stub} with 128 num_context"
        eip_score_1024 = f"{_eip_stub} with 1024 num_context"
        eip_score_7500 = f"{_eip_stub} with 7500 num_context"
        eip_score_10000 = f"{_eip_stub} with 10000 num_context"

        eip_acc_by_ctx = [
            eip_acc_128,
            eip_acc_1024,
            eip_acc_7500,
            eip_acc_10000,
        ]

        eip_mse_by_ctx = [
            eip_mse_128,
            eip_mse_1024,
            eip_mse_7500,
        ]

        eip_score_by_ctx = [
            eip_score_128,
            eip_score_1024,
            eip_score_7500,
            eip_score_10000,
        ]

        t_cross_entropy = "training/CrossEntropyLoss"
        t_huber = "training/HuberLoss"
        e_huber = "evaluation/HuberLoss"

    class OfflineEval:
        eip_score = "Improvement probability over TabPFNv2 (tabpfn-v2)/main score"  # noqa: E501
        eip_score_128 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score with 128 num_context"  # noqa: E501
        eip_score_1024 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score with 1024 num_context"  # noqa: E501
        eip_score_7500 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score with 7500 num_context"  # noqa: E501
        eip_score_10000 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score with 10000 num_context"  # noqa: E501
        eip_acc = (  # noqa: E501
            "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - accuracy"  # noqa: E501
        )  # noqa: E501
        eip_acc_128 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - accuracy with 128 num_context"  # noqa: E501
        eip_acc_1024 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - accuracy with 1024 num_context"  # noqa: E501
        eip_acc_7500 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - accuracy with 7500 num_context"  # noqa: E501
        eip_acc_10000 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - accuracy with 10000 num_context"  # noqa: E501
        eip_mse = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - mse"  # noqa: E501
        eip_mse_128 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - mse with 128 num_context"  # noqa: E501
        eip_mse_1024 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - mse with 1024 num_context"  # noqa: E501
        eip_mse_7500 = "Improvement probability over TabPFNv2 (tabpfn-v2)/score at  - mse with 7500 num_context"  # noqa: E501

        eip_acc_by_ctx = [
            eip_acc_128,
            eip_acc_1024,
            eip_acc_7500,
            eip_acc_10000,
        ]

        eip_mse_by_ctx = [
            eip_mse_128,
            eip_mse_1024,
            eip_mse_7500,
        ]

        eip_score_by_ctx = [
            eip_score_128,
            eip_score_1024,
            eip_score_7500,
            eip_score_10000,
        ]
