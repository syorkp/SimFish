from Configurations.Templates.interruptions import energy_state_profile_long_term

# Ablation configs
with open('Configurations/Run-Configurations/Ablation-Matrices/post_ablation_weights_1_dqn_26_2.npy', 'rb') as f:
    ablation_matrix = np.load(f)
with open('Configurations/Run-Configurations/Ablation-Matrices/post_ablation_weights_2_dqn_26_2.npy', 'rb') as f:
    full_ablation_matrix = np.load(f)


# ========== Training Mode ==========

dqn_training_example = [
    {
        "Model Name": "dqn_training_example",
        "Environment Name": "dqn_example_configuration",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Learning Algorithm": "DQN",
    },
    {
        "Model Name": "dqn_training_example",
        "Environment Name": "dqn_example_configuration",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Learning Algorithm": "DQN",
    },
    {
        "Model Name": "dqn_training_example",
        "Environment Name": "dqn_example_configuration",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Learning Algorithm": "DQN",
    },
    {
        "Model Name": "dqn_training_example",
        "Environment Name": "dqn_example_configuration",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Learning Algorithm": "DQN",
    },
]

ppo_training_example = [
    {
        "Model Name": "ppo_training_example",
        "Environment Name": "ppo_example_configuration",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Learning Algorithm": "PPO",
    },
    {
        "Model Name": "ppo_training_example",
        "Environment Name": "ppo_example_configuration",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Learning Algorithm": "PPO",
    },
    {
        "Model Name": "ppo_training_example",
        "Environment Name": "ppo_example_configuration",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Learning Algorithm": "PPO",
    },
    {
        "Model Name": "ppo_training_example",
        "Environment Name": "ppo_example_configuration",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Learning Algorithm": "PPO",
    },
]

# ========== Assay Modes ==========

# Basic Assay (Naturalistic)

dqn_basic_assay_example = [
    {
        "Model Name": "dqn_salt_only_reduced",
        "Environment Name": "dqn_sor",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 1,
        "Checkpoint": 67,
        "Run Mode": "Assay",
        "Learning Algorithm": "DQN",
        "behavioural recordings": ["environmental positions"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic",
                "repeats": 2,
                "stimulus paradigm": "Naturalistic",
                "duration": 100,
                "tethered": False,
                "save frames": False,
            },
        ]
    },
]

ppo_basic_assay_example = [
    {
        "Model Name": "ppo_proj_reduced",
        "Environment Name": "ppo_proj_1",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Learning Algorithm": "PPO",
        "behavioural recordings": ["environmental positions"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic",
                "repeats": 2,
                "stimulus paradigm": "Naturalistic",
                "duration": 100,
                "tethered": False,
                "save frames": False,
                "use_mu": True,
            },
        ]
    },
]

# Basic Assay (Controlled Stimulus)

dqn_controlled_stimulus_assay_example = [
    {
        "Model Name": "dqn_0_0",
        "Environment Name": "dqn_0_1",
        "Trial Number": 1,
        "Assay Configuration Name": "Controlled-Visual-Stimuli",
        "Run Mode": "Assay",
        "Learning Algorithm": "DQN",
        "behavioural recordings": ["environmental positions"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Moving-Prey",
                "repeats": 1,
                "stimulus paradigm": "Projection",
                "duration": 241,
                "tethered": True,
                "save frames": True,
                "set positions": True,
                "random positions": False,
                "reset": False,
                "reset interval": 1000,
                "moving": True,
                "collisions": True,
                "stimuli": {
                    "prey 1": [
                        {"step": 0,
                         "position": [150, 150]},
                        {"step": 40,
                         "position": [450, 150]},
                        {"step": 80,
                         "position": [450, 450]},
                        {"step": 120,
                         "position": [150, 450]},
                        {"step": 160,
                         "position": [450, 450]},
                        {"step": 200,
                         "position": [450, 150]},
                        {"step": 240,
                         "position": [150, 150]},
                    ],
                },
            },
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

# Basic Assay (to create a VRV producing config

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

# Basic Assay (Interventions)

dqn_interventions_assay_example = [
    {
        "Model Name": "dqn_scaffold_18",
        "Environment Name": "dqn_18_1",
        "Assay Configuration Name": "Behavioural-Data-Full-Interruptions",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Learning Algorithm": "DQN",
        "behavioural recordings": ["environmental positions"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic-1",
                "repeats": 2,
                "stimulus paradigm": "Naturalistic",
                "duration": 2000,
                "tethered": False,
                "save frames": False,
                "interventions": {"visual_interruptions": long_term_interruption_profile,
                                  "preset_energy_state": energy_state_profile_long_term,
                                  "efference_copy_interruptions": long_term_interruption_profile,
                                  "salt_interruptions": long_term_interruption_profile,
                                  "in_light_interruptions": long_term_interruption_profile,
                                  }
            },
        ]
    },
]

dqn_new_data_rnn_manipulations = [
    {
        "Model Name": "dqn_new",
        "Environment Name": "dqn_new_1",
        "Assay Configuration Name": "Behavioural-Data-RNN-Zero",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Learning Algorithm": "DQN",
        "set random seed": True,
        "behavioural recordings": ["environmental positions"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic",
                "repeats": 1,
                "stimulus paradigm": "Naturalistic",
                "duration": 10000,
                "tethered": False,
                "save frames": False,
                "use_mu": True,
                "interventions": {"rnn_input": rnn_zeros}
            },
        ],
    },
    {
        "Model Name": "dqn_new",
        "Environment Name": "dqn_new_2",
        "Assay Configuration Name": "Behavioural-Data-RNN-Zero",
        "Trial Number": 2,
        "Run Mode": "Assay",
        "Learning Algorithm": "DQN",
        "Assays": [
            {
                "assay id": "Naturalistic",
                "repeats": 1,
                "stimulus paradigm": "Naturalistic",
                "duration": 10000,
                "tethered": False,
                "save frames": False,
                "use_mu": True,
                "behavioural recordings": ["environmental positions"],
                "network recordings": ["rnn_shared", "internal_state"],
                "interventions": {"rnn_input": rnn_zeros}
            },
        ],
    },

]

# Basic Assay (Ablations)


# Split Assay

dqn_split_assay_example = [
    {
        "Model Name": "dqn_0_0",
        "Environment Name": "dqn_0_1",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 1,
        "Run Mode": "Split-Assay",
        "Split Event": "One-Prey-Close",
        "Modification": "Nearby-Prey-Removal",
        "Learning Algorithm": "DQN",
        "behavioural recordings": ["environmental positions", "observation", "reward assessments"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic",
                "stimulus paradigm": "Naturalistic",
                "repeats": 1,
                "duration": 500,
                "tethered": False,
                "save frames": False,
            },
        ]
    }
]

ppo_split_assay_example = [
    {
        "Model Name": "ppo_proj",
        "Environment Name": "ppo_proj_1",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 1,
        "Run Mode": "Split-Assay",
        "Split Event": "One-Prey-Close",
        "Modification": "Nearby-Prey-Removal",
        "Learning Algorithm": "PPO",
        "behavioural recordings": ["environmental positions"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic",
                "stimulus paradigm": "Naturalistic",
                "repeats": 1,
                "duration": 500,
                "tethered": False,
                "save frames": False,
                "use_mu": True,
            },
        ]
    }
]

# Assay-Analysis-Across-Scaffold

dqn_analysis_across_scaffold_example = [
    {
        "Model Name": "dqn_0_0",
        "Environment Name": "dqn_0_1",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Config Modification": "Empty",
        "Trial Number": 1,
        "Delete Data": False,
        "Run Mode": "Assay-Analysis-Across-Scaffold",
        "Learning Algorithm": "DQN",
        "behavioural recordings": ["environmental positions"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic",
                "repeats": 1,
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "tethered": False,
                "save frames": False,
            },
        ],
        "Analysis": [
            {
                "analysis id": "Turn-Analysis",
                "analysis script": "Analysis.Behavioural.Exploration.turning_analysis_discrete",
                "analysis function": "plot_all_turn_analysis",
                "analysis arguments": ["model_name", "assay_config_name", "Naturalistic", 1],
            }
        ],
    }
]

ppo_analysis_across_scaffold_example = [
    {
        "Model Name": "ppo_proj_1",
        "Environment Name": "ppo_proj_1",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Config Modification": "Empty",
        "Trial Number": 1,
        "Run Mode": "Assay-Analysis-Across-Scaffold",
        "Learning Algorithm": "PPO",
        "behavioural recordings": ["environmental positions", "observation"],
        "network recordings": ["rnn_shared", "internal_state"],
        "Assays": [
            {
                "assay id": "Naturalistic",
                "repeats": 1,
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "tethered": False,
                "save frames": False,
                "use_mu": True,
            },
        ],
        "Analysis": [
            {
                "analysis id": "Turn-Analysis",
                "analysis script": "Analysis.Behavioural.Exploration.turning_analysis_continuous",
                "analysis function": "plot_all_turn_analysis_continuous",
                "analysis arguments": ["model_name", "assay_config_name", "Naturalistic", 1],
            }
        ],
    }
]
