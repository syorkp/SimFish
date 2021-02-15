import os
from datetime import datetime

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")


current_assay_configuration = [
    {
        "Model Name": "changed_penalties",
        "Environment Name": "prey_only",
        "Assay Configuration Name": "Naturalistic_test",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Assays": [
            {
                "assay id": "Naturalistic-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 100,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "reset": False,
                "recordings": ["observation", "behavioural choice", "rnn 2 state"],
                "interactions": []
            },
            # {
            #     "assay id": "Naturalistic-2",
            #     "stimulus paradigm": "Naturalistic",
            #     "duration": 1000,
            #     "Tethered": False,
            #     "save frames": True,
            #     "random positions": False,
            #     "reset": False,
            #     "recordings": ["observation", "behavioural choice"],
            #     "interactions": []
            # },
        ],
    },
]

current_training_configuration = [
    {
        "Model Name": "large_all_features",
        "Environment Name": "large_all_features",
        "Trial Number": 1,
        "Total Configurations": 1,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "normal_all_features",
        "Environment Name": "normal_all_features",
        "Trial Number": 1,
        "Total Configurations": 1,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "extra_layer_all_features",
        "Environment Name": "extra_layer_all_features",
        "Trial Number": 1,
        "Total Configurations": 1,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(current_assay_configuration)
manager.run_priority_loop()
