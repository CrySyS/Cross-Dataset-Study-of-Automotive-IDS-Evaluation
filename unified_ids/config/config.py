
"""
Evaluation configuration: which methods run on which datasets, and
some method grouping metadata.
"""


EVAL_PLAN_all_3_method_on_all_3_dataset = [
        {
            "train_glob": "data_parquet/03_Car-HackingDataset/normal_run_data.parquet",
            "test_glob": "data_parquet/03_Car-HackingDataset/*attack.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "train_glob": "data_parquet/04_CAN-IntrusionDataset_OTIDS/train/*.parquet",
            "test_glob": "data_parquet/04_CAN-IntrusionDataset_OTIDS/test/*.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "DAGA_STABILI2022_ArbitrarySequenceReplay",
            "train_glob": "data_parquet/05_DAGA_STABILI2022/NormalData/*.parquet",
            "test_glob": "data_parquet/05_DAGA_STABILI2022/Attack_ArbitrarySequenceReplay/infected/*.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "DAGA_STABILI2022_DenialOfService",
            "train_glob": "data_parquet/05_DAGA_STABILI2022/NormalData/*.parquet",
            "test_glob": "data_parquet/05_DAGA_STABILI2022/Attack_DenialOfService/infected/*.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "DAGA_STABILI2022_MessageIDFuzzing",
            "train_glob": "data_parquet/05_DAGA_STABILI2022/NormalData/*.parquet",
            "test_glob": "data_parquet/05_DAGA_STABILI2022/Attack_MessageIDFuzzing/infected/*.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "DAGA_STABILI2022_OrderedSequenceReplay",
            "train_glob": "data_parquet/05_DAGA_STABILI2022/NormalData/*.parquet",
            "test_glob": "data_parquet/05_DAGA_STABILI2022/Attack_OrderedSequenceReplay/infected/*.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
    ]


EVAL_PLAN_CIDv2 = [
        {
            "name": "CIDv2_OpelAstra_all",
            "train_glob": "data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/training.parquet",
            "test_glob": [
                "data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/testing.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/diagnostic.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/dosattack.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/fuzzing_canid.parquet",
                #"data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/fuzzing_payload.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/replay.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/OpelAstra/suspension.parquet",
            ],
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "CIDv2_Prototype_all",
            "train_glob": "data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/training.parquet",
            "test_glob": [
                "data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/testing.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/diagnostic.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/dosattack.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/fuzzing_canid.parquet",
                #"data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/fuzzing_payload.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/spoofing_speedometer.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/Prototype/suspension.parquet",
            ],
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "CIDv2_RenaultClio_all",
            "train_glob": "data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/training.parquet",
            "test_glob": [
                "data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/testing.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/diagnostic.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/dosattack.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/fuzzing_canid.parquet",
                #"data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/fuzzing_payload.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/replay.parquet",
                "data_parquet/07_CAN_Intrusion_Dataset_v2/RenaultClio/suspension.parquet",
            ],
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
    ]


EVAL_PLAN_CrySyS_injection_chunked = [
        {
            "name": "CrySyS_injection_S",
            "train_glob": "data_parquet/06_CrySyS/S*/*-benign.parquet",
            "test_glob": [
                "data_parquet/06_CrySyS/S*/*-msg-inj-*[0-9].parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "CrySyS_injection_T",
            "train_glob": "data_parquet/06_CrySyS/T*/*-benign.parquet",
            "test_glob": [
                "data_parquet/06_CrySyS/T*/*-msg-inj-*[0-9].parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
    ]
EVAL_PLAN_CrySyS_and_Road = [
        {
            "name": "CrySyS_injection",
            "train_glob": "data_parquet/06_CrySyS/*/*-benign.parquet",
            "test_glob": [
                "data_parquet/06_CrySyS/*/*-msg-inj-*[0-9].parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "ROAD_injection",
            "train_glob": [
                "data_parquet/02_Road/injection_nonmodified/ambient/*.parquet",
            ],
            "test_glob": [
                "data_parquet/02_Road/injection_nonmodified/attacks/*.parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
    ]


EVAL_PLAN_CrySyS_and_Road_subset_balanced_DONE = [
        {
            "name": "CrySyS_injection_S_all",
            "train_glob": "data_parquet/06_CrySyS/S*/*-benign.parquet",
            "test_glob": [
                "data_parquet/06_CrySyS/S*/*-msg-inj-*[0-9].parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
                {
            "name": "CrySyS_injection_T_heavy_1_2",
            "train_glob": "data_parquet/06_CrySyS/T-1-[1-2]/*-benign.parquet",
            "test_glob": [
                "data_parquet/06_CrySyS/T-1-[1-2]/*-msg-inj-*[0-9].parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        }]
EVAL_PLAN_CrySyS_and_Road_subset_balanced = [

                {
            "name": "CrySyS_injection_T_heavy_3_4",
            "train_glob": "data_parquet/06_CrySyS/T-1-[3-4]/*-benign.parquet",
            "test_glob": [
                "data_parquet/06_CrySyS/T-1-[3-4]/*-msg-inj-*[0-9].parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
                {
            "name": "CrySyS_injection_T_heavy_5_6",
            "train_glob": "data_parquet/06_CrySyS/T-1-[5-6]/*-benign.parquet",
            "test_glob": [
                "data_parquet/06_CrySyS/T-1-[5-6]/*-msg-inj-*[0-9].parquet"
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "CrySyS_injection_T_heavy_B",
            "train_glob": [
                "data_parquet/06_CrySyS/T-1-7/*-benign.parquet",
                "data_parquet/06_CrySyS/T-2-*/*-benign.parquet",
                "data_parquet/06_CrySyS/T-3-*/*-benign.parquet",
            ],
            "test_glob": [
                "data_parquet/06_CrySyS/T-1-7/*-msg-inj-*[0-9].parquet",
                "data_parquet/06_CrySyS/T-2-*/*-msg-inj-*[0-9].parquet",
                "data_parquet/06_CrySyS/T-3-*/*-msg-inj-*[0-9].parquet",
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "ROAD_injection_heavy",
            "train_glob": [
                "data_parquet/02_Road/injection_nonmodified/ambient/*.parquet",
            ],
            "test_glob": [
                "data_parquet/02_Road/injection_nonmodified/attacks/accelerator_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_speedometer_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_on_attack_*.parquet",
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "name": "ROAD_injection_light",
            "train_glob": [
                "data_parquet/02_Road/injection_nonmodified/ambient/*.parquet",
            ],
            "test_glob": [
                "data_parquet/02_Road/injection_nonmodified/attacks/correlated_signal_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/fuzzing_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_engine_coolant_temp_attack.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_off_attack_*.parquet",
            ],
            "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
    ]

EVAL_PLAN_ROAD_quick_representative = [
        {
            "name": "ROAD_sub_mba",
            "train_glob": [
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_basic_short.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_extended_short.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_benign_anomaly.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_idle_radio_infotainment.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_highway_street_driving_long.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_highway_street_driving_diagnostics.parquet",
            ],
            "test_glob": [
                "data_parquet/02_Road/injection_nonmodified/attacks/correlated_signal_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/fuzzing_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_engine_coolant_temp_attack.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_speedometer_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_off_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_on_attack_*.parquet",
            ],
            "methods": ["mba_ocsvm_v2"],
        }]
EVAL_PLAN_ROAD_quick_representative_DONE = [
                {
            "name": "ROAD_sub_daga",
            "train_glob": [
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_basic_short.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_extended_short.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_benign_anomaly.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_idle_radio_infotainment.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_highway_street_driving_long.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_highway_street_driving_diagnostics.parquet",
            ],
            "test_glob": [
                "data_parquet/02_Road/injection_nonmodified/attacks/correlated_signal_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/fuzzing_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_engine_coolant_temp_attack.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_speedometer_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_off_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_on_attack_*.parquet",
            ],
            "methods": ["daga_ngram"],
        },
                {
            "name": "ROAD_sub_assocrules",
            "train_glob": [
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_basic_short.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_extended_short.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_drive_benign_anomaly.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_dyno_idle_radio_infotainment.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_highway_street_driving_long.parquet",
                "data_parquet/02_Road/injection_nonmodified/ambient/ambient_highway_street_driving_diagnostics.parquet",
            ],
            "test_glob": [
                "data_parquet/02_Road/injection_nonmodified/attacks/correlated_signal_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/fuzzing_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_engine_coolant_temp_attack.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/max_speedometer_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_off_attack_*.parquet",
                "data_parquet/02_Road/injection_nonmodified/attacks/reverse_light_on_attack_*.parquet",
            ],
            "methods": [ "assoc_rules"],
        },
    ]

EVAL_PLAN_DAGA_PATCHED = [
    {
        "name": "DAGA_STABILI2022_ArbitrarySequenceReplay",
        "train_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/clean/*.parquet",
        "test_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/infected/ArbitrarySequenceReplay__*.parquet",
        "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
    },
    #{ DONE
    #    "name": "DAGA_STABILI2022_DenialOfService",
    #    "train_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/clean/*.parquet",
    #    "test_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/infected/DenialOfService__*.parquet",
    #    "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
    #},
    #{
    #    "name": "DAGA_STABILI2022_MessageIDFuzzing",
    #    "train_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/clean/*.parquet",
    #    "test_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/infected/MessageIDFuzzing__*.parquet",
    #    "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
    #},
    {
        "name": "DAGA_STABILI2022_OrderedSequenceReplay",
        "train_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/clean/*.parquet",
        "test_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/infected/OrderedSequenceReplay__*.parquet",
        "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
    },
        {
        "name": "DAGA_STABILI2022_OrderedSequenceReplay",
        "train_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/clean/*.parquet",
        "test_glob": "data_parquet/05_DAGA_STABILI2022/_rebuild_check_no_payload/infected/SingleIDReplay__*.parquet",
        "methods": ["mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
    }
]

EVAL_PLAN_CH_OTIDS_rerun = [
        {
            "train_glob": "data_parquet/03_Car-HackingDataset/normal_run_data.parquet",
            "test_glob": "data_parquet/03_Car-HackingDataset/*attack.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        },
        {
            "train_glob": "data_parquet/04_CAN-IntrusionDataset_OTIDS/train/*.parquet",
            "test_glob": "data_parquet/04_CAN-IntrusionDataset_OTIDS/test/*.parquet",
            "methods": [ "mba_ocsvm_v2", "daga_ngram", "assoc_rules"],
        }]
EVAL_PLAN = EVAL_PLAN_ROAD_quick_representative