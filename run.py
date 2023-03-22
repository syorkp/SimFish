import os
import sys

import json
from datetime import datetime
import numpy as np

from Services.trial_manager import TrialManager
from Configurations.Templates.interruptions import energy_state_profile_long_term, salt_profile_long_term, \
    in_light_profile_long_term, efference_A, efference_B, efference_C, efference_D, efference_E, efference_F, \
    efference_G, efference_H, efference_V, efference_W, efference_X, efference_Y, efference_Z, long_term_visual_interruption_profile, \
    brief_interruption_profile, fish_relocation_to_nowhere, energy_state_profile_long_term, long_term_interruption_profile, rnn_zeros

if __name__ == "__main__":  # may be needed to run on windows
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

    with open("./Run-Configurations/VRV_CONFIG.json", "r") as file:
        vrv_config = json.load(file)

    vrv_config = vrv_config[0]
    vrv_config["Model Name"] = "dqn_scaffold_14"
    vrv_config["Trial Number"] = 1
    vrv_config["Checkpoint"] = None
    vrv_config["New Simulation"] = True
    vrv_config["Full Reafference"] = False
    for i, assay in enumerate(vrv_config["Assays"]):
        vrv_config["Assays"][i]["behavioural recordings"] = ["environmental positions", "observation"]
        vrv_config["Assays"][i]["network recordings"] = ["rnn state", "internal state"]
        vrv_config["Assays"][i]["use_mu"] = True
        vrv_config["Assays"][i]["energy_state_control"] = "Held"
        vrv_config["Assays"][i]["salt_control"] = False
    vrv_config["Using GPU"] = False
    vrv_config["Continuous Actions"] = False
    vrv_config["SB Emulator"] = True
    vrv_config["Learning Algorithm"] = "DQN"
    vrv_config["Assays"] = [vrv_config["Assays"][0]]
    vrv_config["Assays"][0]["save frames"] = True
    vrv_config["Assays"][0]["stimuli"]["prey 1"]["steps"] = 200
    vrv_config["Assays"][0]["duration"] = 200

    vrv_config = [vrv_config]

    # Ablation configs
    with open('Configurations/Ablation-Matrices/post_ablation_weights_1_dqn_26_2.npy', 'rb') as f:
        ablation_matrix = np.load(f)
    with open('Configurations/Ablation-Matrices/post_ablation_weights_2_dqn_26_2.npy', 'rb') as f:
        full_ablation_matrix = np.load(f)


    #                    ASSAY

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
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
                ]
        }
        ]

    dqn_free_config_large_gamma_1 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_final",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Trial Number": 1,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Using GPU": False,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 100,
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        }
        ]

    dqn_free_config_large_gamma_3 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_final_x",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Trial Number": 1,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 50,
                    "duration": 100,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        }
        ]

    assay_gathering_test = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_final",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Trial Number": 3,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Using GPU": False,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 1,
                    "duration": 10,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        }
        ]


    # Split timelines assay

    dqn_split_assay_test = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_final_mod",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Trial Number": 3,
            "Run Mode": "Split-Assay",
            "Split Event": "One-Prey-Close",
            "Modification": "Nearby-Prey-Removal",
            "Learning Algorithm": "DQN",
            "Using GPU": False,
            "behavioural recordings": ["environmental positions", "observation", "reward assessments"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 1,
                    "duration": 2000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        }
    ]

    ppo_split_assay_test = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma_2",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Trial Number": 2,
            "Run Mode": "Split-Assay",
            "Split Event": "One-Prey-Close",
            "Modification": "Nearby-Prey-Removal",
            "Learning Algorithm": "PPO",
            "behavioural recordings": ["environmental positions", "observation", "reward assessments"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 1,
                    "duration": 2000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        }
    ]

    # Assay And Analysis

    dqn_gamma_analysis_across_scaffold_1 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_free",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Config Modification": "Empty",
            "Trial Number": 1,
            "Delete Data": True,
            "Run Mode": "Assay-Analysis-Across-Scaffold",
            "Learning Algorithm": "DQN",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "repeats": 10,
                    "stimulus paradigm": "Naturalistic",
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ],
            "Analysis": [
                {
                    "analysis id": "Turn-Analysis",
                    "analysis script": "Analysis.Behavioural.Exploration.turning_analysis_discrete",
                    "analysis function": "plot_all_turn_analysis",
                    "analysis arguments": ["model_name", "assay_config_name", "Naturalistic", 10],
                    "Delete Data": True
                }
            ],
        }
    ]

    ppo_gamma_analysis_across_scaffold_1 = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma_free",
            "Assay Configuration Name": "Behavioural-Data-Free",
            "Config Modification": "Empty",
            "Trial Number": 3,
            "Run Mode": "Assay-Analysis-Across-Scaffold",
            "Learning Algorithm": "PPO",
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "repeats": 1,
                    "stimulus paradigm": "Naturalistic",
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ],
            "Analysis": [
                {
                    "analysis id": "Turn-Analysis",
                    "analysis script": "Analysis.Behavioural.Exploration.turning_analysis_continuous",
                    "analysis function": "plot_all_turn_analysis_continuous",
                    "analysis arguments": ["model_name", "assay_config_name", "Naturalistic", 1],
                    "Delete Data": True
                }
            ],
        }
    ]

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

    dqn_test_1 = [
        {
            "Model Name": "dqn_test",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Profile Speed": True,
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_test_scaled_1 = [
        {
            "Model Name": "dqn_test_scaled",
            "Environment Name": "dqn_epsilon_scaled",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Profile Speed": True,
            "Learning Algorithm": "DQN",
        },
    ]


    dqn_epsilon_1 = [
        {
            "Model Name": "dqn_epsilon",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_epsilon_2 = [
        {
            "Model Name": "dqn_epsilon",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_epsilon_3 = [
        {
            "Model Name": "dqn_epsilon",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_epsilon_4 = [
        {
            "Model Name": "dqn_epsilon",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_epsilon_5 = [
        {
            "Model Name": "dqn_epsilon",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 5,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_gamma_sg_1 = [
        {
            "Model Name": "dqn_gamma_sg",
            "Environment Name": "dqn_gamma_sg",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_gamma_sg_2 = [
        {
            "Model Name": "dqn_gamma_sg",
            "Environment Name": "dqn_gamma_sg",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_gamma_pm_1 = [
        {
            "Model Name": "dqn_gamma_pm",
            "Environment Name": "dqn_gamma_pm",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_gamma_pm_2 = [
        {
            "Model Name": "dqn_gamma_pm",
            "Environment Name": "dqn_gamma_pm",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_gamma_pm_3 = [
        {
            "Model Name": "dqn_gamma_pm",
            "Environment Name": "dqn_gamma_pm",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_gamma_pm_4 = [
        {
            "Model Name": "dqn_gamma_pm",
            "Environment Name": "dqn_gamma_pm",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_gamma_pm_5 = [
        {
            "Model Name": "dqn_gamma_pm",
            "Environment Name": "dqn_gamma_pm",
            "Trial Number": 5,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_gamma_6 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma",
            "Trial Number": 6,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_gamma_7 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma",
            "Trial Number": 7,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_predator_1 = [
        {
            "Model Name": "dqn_predator",
            "Environment Name": "dqn_predator",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_predator_2 = [
        {
            "Model Name": "dqn_predator",
            "Environment Name": "dqn_predator",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_delta_ns_1 = [
        {
            "Model Name": "dqn_delta_ns",
            "Environment Name": "dqn_delta_ns",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_delta_ns_2 = [
        {
            "Model Name": "dqn_delta_ns",
            "Environment Name": "dqn_delta_ns",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_delta_1 = [
        {
            "Model Name": "dqn_delta",
            "Environment Name": "dqn_delta",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_delta_2 = [
        {
            "Model Name": "dqn_delta",
            "Environment Name": "dqn_delta",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_delta_3 = [
        {
            "Model Name": "dqn_delta",
            "Environment Name": "dqn_delta",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_delta_4 = [
        {
            "Model Name": "dqn_delta",
            "Environment Name": "dqn_delta",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_delta_5 = [
        {
            "Model Name": "dqn_delta",
            "Environment Name": "dqn_delta",
            "Trial Number": 5,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_delta_test_1 = [
        {
            "Model Name": "dqn_delta_test",
            "Environment Name": "dqn_delta_test",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    #                   TRAINING - PPO

    ppo_gamma_1 = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
    ]
    ppo_gamma_2 = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
    ]
    ppo_gamma_3 = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
    ]
    ppo_gamma_4 = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
    ]
    ppo_gamma_5 = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma",
            "Trial Number": 5,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },

    ]
    ppo_gamma_6 = [
        {
            "Model Name": "ppo_gamma",
            "Environment Name": "ppo_gamma",
            "Trial Number": 6,
            "Using GPU": True,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },

    ]

    #                       Testing Robustness of master branch
    # Last used config (reaffirm base test).
    dqn_epsilon_6 = [
        {
            "Model Name": "dqn_epsilon",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 6,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    dqn_epsilon_7 = [
        {
            "Model Name": "dqn_epsilon",
            "Environment Name": "dqn_epsilon",
            "Trial Number": 7,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    # Different reward balance
    dqn_epsilon_rb = [
        {
            "Model Name": "dqn_epsilon_rb",
            "Environment Name": "dqn_epsilon_rb",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    # Different reward scale
    dqn_epsilon_rs = [
        {
            "Model Name": "dqn_epsilon_rs",
            "Environment Name": "dqn_epsilon_rs",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    # Rare predators
    dqn_epsilon_rp = [
        {
            "Model Name": "dqn_epsilon_rp",
            "Environment Name": "dqn_epsilon_rp",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    # Gathering for lab meeting
    dqn_lab_meeting_gathering_1 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_3",
            "Assay Configuration Name": "Behavioural-Data-Free-A",
            "Trial Number": 3,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Checkpoint": 11609,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 100,
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        },
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_5",
            "Assay Configuration Name": "Behavioural-Data-Free-A",
            "Trial Number": 5,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Checkpoint": 4016,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 100,
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        }
        ]
    dqn_lab_meeting_gathering_2 = [
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_s1",
            "Assay Configuration Name": "Behavioural-Data-Free-A",
            "Trial Number": 4,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Checkpoint": 2012,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 100,
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        },
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_s27",
            "Assay Configuration Name": "Behavioural-Data-Free-B",
            "Trial Number": 4,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Checkpoint": 4002,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 100,
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        },
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_s51",
            "Assay Configuration Name": "Behavioural-Data-Free-C",
            "Trial Number": 4,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Checkpoint": 6055,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 100,
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        },
        {
            "Model Name": "dqn_gamma",
            "Environment Name": "dqn_gamma_3",
            "Assay Configuration Name": "Behavioural-Data-Free-D",
            "Trial Number": 4,
            "Run Mode": "Assay",
            "Learning Algorithm": "DQN",
            "Checkpoint": 12625,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": ["rnn_shared", "internal_state"],
            "Assays": [
                {
                    "assay id": "Naturalistic",
                    "stimulus paradigm": "Naturalistic",
                    "repeats": 100,
                    "duration": 10000,
                    "Tethered": False,
                    "save frames": False,
                    "use_mu": True,
                    "save stimuli": False,
                    "random positions": False,
                    "reset": False,
                    "background": None,
                    "moving": False,
                    "collisions": True,
                },
            ]
        },
        ]


    # DEBUGGING (20.03 onwards)
    dqn_basic_1 = [
        {
            "Model Name": "dqn_epsilon_static",
            "Environment Name": "dqn_epsilon_static",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_epsilon_static",
            "Environment Name": "dqn_epsilon_static",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_epsilon_static",
            "Environment Name": "dqn_epsilon_static",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_epsilon_static",
            "Environment Name": "dqn_epsilon_static",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]


    if run_config is None:
        run_config = dqn_basic_1
    else:
        print(f"{run_config} entered.")
        run_config = globals()[run_config]

    print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    manager = TrialManager(run_config, parallel_jobs=5)
    manager.run_priority_loop()
