import os
import sys

import json
from datetime import datetime
import numpy as np

from Services.trial_manager import TrialManager

if __name__ == "__main__": # may be needed to run on windows
    # Get config argument
    try:
        run_config = sys.argv[1]
    except IndexError:
        run_config = None

    # Ensure output directories exist
    if not os.path.exists("./Training-Output/"):
        os.makedirs("./Training-Output/")

    if not os.path.exists("./Assay-Output/"):
        os.makedirs("./Assay-Output/")

    # Setting .nv location to prevent GPU error
    # if not os.path.exists("./GPU-Caches/"):
    #     os.makedirs("./GPU-Caches/")
    #
    # directory = "./GPU-Caches/"
    # existing_caches = [os.path.join(o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))]
    #
    # if len(existing_caches) > 0:
    #     caches_as_int = [int(o) for o in existing_caches]
    #     last_cache_number = max(caches_as_int)
    #     os.makedirs(f"./GPU-Caches/{last_cache_number + 1}")
    #     os.environ["__GL_SHADER_DISK_CACHE_PATH"] = f"./GPU-Caches/{last_cache_number+1}/"
    # else:
    #     os.makedirs(f"./GPU-Caches/{1}")
    #     os.environ["__GL_SHADER_DISK_CACHE_PATH"] = f"./GPU-Caches/{1}/"


    # Loading VRV configs

    with open("Configurations/Run-Configurations/VRV_CONFIG.json", "r") as file:
        vrv_config = json.load(file)

    vrv_config = vrv_config[0]
    vrv_config["Model Name"] = "dqn_scaffold_14"
    vrv_config["Trial Number"] = 1
    for i, assay in enumerate(vrv_config["Assays"]):
        vrv_config["Assays"][i]["behavioural recordings"] = ["environmental positions", "observation"]
        vrv_config["Assays"][i]["network recordings"] = ["rnn state", "internal state"]
        vrv_config["Assays"][i]["use_mu"] = True
        vrv_config["Assays"][i]["energy_state_control"] = "Held"
        vrv_config["Assays"][i]["salt_control"] = False
    vrv_config["Learning Algorithm"] = "DQN"
    vrv_config["Assays"] = [vrv_config["Assays"][0]]
    vrv_config["Assays"][0]["save frames"] = True
    vrv_config["Assays"][0]["stimuli"]["prey 1"]["steps"] = 200
    vrv_config["Assays"][0]["duration"] = 200

    vrv_config = [vrv_config]

    # Ablation configs
    with open('Configurations/Run-Configurations/Ablation-Matrices/post_ablation_weights_1_dqn_26_2.npy', 'rb') as f:
        ablation_matrix = np.load(f)
    with open('Configurations/Run-Configurations/Ablation-Matrices/post_ablation_weights_2_dqn_26_2.npy', 'rb') as f:
        full_ablation_matrix = np.load(f)

    # Config Examples

    controlled_assay_configuration_2 = [
        {
            "Model Name": "dqn_0_0",
            "Environment Name": "dqn_0_1",
            "Trial Number": 1,
            "Assay Configuration Name": "Controlled-Visual-Stimuli",
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                # {
                #     "assay id": "Moving-Prey",
                #     "repeats": 1,
                #     "stimulus paradigm": "Projection",
                #     "duration": 241,
                #     "tethered": True,
                #     "save frames": True,
                #     "set positions": True,
                #     "random positions": False,
                #     "reset": False,
                #     "reset interval": 1000,
                #     "moving": True,
                #     "collisions": True,
                #     "stimuli": {
                #         "prey 1": [
                #             {"step": 0,
                #              "position": [150, 150]},
                #             {"step": 40,
                #              "position": [450, 150]},
                #             {"step": 80,
                #              "position": [450, 450]},
                #             {"step": 120,
                #              "position": [150, 450]},
                #             {"step": 160,
                #              "position": [450, 450]},
                #             {"step": 200,
                #              "position": [450, 150]},
                #             {"step": 240,
                #              "position": [150, 150]},
                #         ],
                #     },
                # },
                {
                    "assay id": "Moving-Predator",
                    "repeats": 1,
                    "stimulus paradigm": "Projection",
                    "duration": 121,
                    "tethered": True,
                    "save frames": True,
                    "set positions": True,
                    "random positions": False,
                    "reset": False,
                    "reset interval": 1000,
                    "moving": True,
                    "collisions": True,
                    "recordings": ["behavioural choice", "rnn state", "observation"],
                    "stimuli": {
                        "predator 1": [
                            {"step": 0,
                             "position": [100, 100]},
                            {"step": 20,
                             "position": [500, 100]},
                            {"step": 40,
                             "position": [500, 500]},
                            {"step": 60,
                             "position": [100, 500]},
                            {"step": 80,
                             "position": [500, 500]},
                            {"step": 100,
                             "position": [500, 100]},
                            {"step": 120,
                             "position": [100, 100]},
                        ],
                    },
                }
            ]
        },
    ]

    #                    ASSAY

    # Assay mode 1: Base

    dqn_empty_config_large_gamma_1 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_empty",
            "Assay Configuration Name": "Behavioural-Data-Empty",
            "Trial Number": 1,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "repeats": 100,
                    "stimulus paradigm": "Naturalistic",
                    "duration": 10000,
                    "tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "random positions": False,
                    "reset": False,
                    "moving": False,
                    "collisions": True,
                },
                ]
        }
        ]

    dqn_0_initial = [
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0_1",
            "Assay Configuration Name": "Behavioural-Data-Empty",
            "Trial Number": 1,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "repeats": 40,
                    "stimulus paradigm": "Naturalistic",
                    "duration": 10000,
                    "tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "random positions": False,
                    "reset": False,
                    "moving": False,
                    "collisions": True,
                },
                ]
        },
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0_7",
            "Assay Configuration Name": "Behavioural-Data-Empty",
            "Trial Number": 7,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "repeats": 40,
                    "stimulus paradigm": "Naturalistic",
                    "duration": 10000,
                    "tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "random positions": False,
                    "reset": False,
                    "moving": False,
                    "collisions": True,
                },
            ]
        }
        ]

    dqn_salt_zeroed_assay = [
        {
            "Model Name": "dqn_salt_only_reduced_z",
            "Environment Name": "dqn_sor",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Trial Number": 1,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "repeats": 40,
                    "stimulus paradigm": "Naturalistic",
                    "duration": 10000,
                    "tethered": False,
                    "save frames": False,
                },
                ]
        },
        ]


    # Assay mode 3: Assay And Analysis



    #                   TRAINING - DQN

    local_test = [
        {
            "Model Name": "local_test",
            "Environment Name": "local_test",
            "Trial Number": 1,
            "Using GPU": False,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    local_test_large = [
        {
            "Model Name": "local_test_large",
            "Environment Name": "local_test_large",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_0 = [
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_0_2 = [
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 5,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 6,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 7,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_0",
            "Environment Name": "dqn_0",
            "Trial Number": 8,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_salt_only = [
        {
            "Model Name": "dqn_salt_only",
            "Environment Name": "dqn_salt_only",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only",
            "Environment Name": "dqn_salt_only",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only",
            "Environment Name": "dqn_salt_only",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only",
            "Environment Name": "dqn_salt_only",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_salt_only_reduced = [
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_salt_only_reduced_2 = [
        {
            "Model Name": "dqn_salt_only_reduced_2",
            "Environment Name": "dqn_salt_only_reduced_2",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced_2",
            "Environment Name": "dqn_salt_only_reduced_2",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced_2",
            "Environment Name": "dqn_salt_only_reduced_2",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced_2",
            "Environment Name": "dqn_salt_only_reduced_2",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    #                   TRAINING - PPO

    ppo_proj = [
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
    ]

    if run_config is None:
        run_config = controlled_assay_configuration_2
    else:
        print(f"{run_config} entered.")
        run_config = globals()[run_config]

    print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    manager = TrialManager(run_config, parallel_jobs=5)
    manager.run_priority_loop()
