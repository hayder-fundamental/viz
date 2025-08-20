class Tags:
    classification = "classification"
    regression = "regression"
    relevant = "Hayder::relevant"

    # TODO(HE): move this w/ refactor
    ms4_sota = "Hayder::MS4-SOTA"


class MetricNames:
    _eip_stub = "evaluation_Improvement probability over TabPFNv2fixed/score at  -"
    # works
    t_cross_entropy = "training/CrossEntropyLoss"

    eip_acc = (
        "evaluation_Improvement probability over TabPFNv2fixed/score at  - accuracy"
    )
    eip_acc_32 = f"{_eip_stub} accuracy with 32 num_context"
    eip_acc_128 = f"{_eip_stub} accuracy with 128 num_context"
    eip_acc_1024 = f"{_eip_stub} accuracy with 1024 num_context"
    eip_acc_7500 = f"{_eip_stub} accuracy with 7500 num_context"

    eip_accs = [
        eip_acc,
        eip_acc_32,
        eip_acc_128,
        eip_acc_1024,
        eip_acc_7500,
    ]

    eip_mse = "evaluation_Improvement probability over TabPFNv2fixed/score at  - mse"
    t_huber = "training/HuberLoss"
    e_huber = "evaluation/HuberLoss"
