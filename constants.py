class MetricNames:
    # works
    t_cross_entropy = "training/CrossEntropyLoss"

    eip_acc = (
        "evaluation_Improvement probability over TabPFNv2fixed/score at  - accuracy"
    )
    eip_acc_32 = "evaluation_Improvement probability over TabPFNv2fixed/score at  - accuracy with 32 num_context"
    eip_acc_128 = "evaluation_Improvement probability over TabPFNv2fixed/score at  - accuracy with 128 num_context"
    eip_acc_1024 = "evaluation_Improvement probability over TabPFNv2fixed/score with at  - accuracy 1024 num_context"
    eip_acc_7500 = "evaluation_Improvement probability over TabPFNv2fixed/score with at  - accuracy 7500 num_context"

    eips = [
        eip_acc,
        eip_acc_32,
        eip_acc_128,
        eip_acc_1024,
        eip_acc_7500,
    ]


class RunIDs:
    woj_tabicl_run_ids = [
        # Simplified Tabicl V0
        "li9vmts8",
        # SimpleTabICL model ctd @lr=1e-4 + rng fix, smaller LR restart
        "4ij90pn6",
    ]
    tabicl_run_ids = [
        "1pmir581",
        "fguyfgu3",
        "91xwo8vq",
        "bk6hy1u8",
        "fipt4khi",
        "fnemi4d4",
        "4qlafc3f",
        "3kc8angw",
        "pkq7w59y",
        "l3vdkvcc",
        "o8dxdwnj",
        "vzjjudsp",
        "f8ly6enm",
        *woj_tabicl_run_ids,
    ]

