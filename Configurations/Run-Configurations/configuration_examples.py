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

# Basic Assay (Controlled Stimulus)  TODO: Create

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
