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
        "Model Name": "larger_network",
        "Environment Name": "prey_only",
        "Assay Configuration Name": "Naturalistic",
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
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "recordings": ["behavioural choice", "position", "fish_angle", "predator_position", "prey_positions"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "recordings": ["behavioural choice", "position", "fish_angle", "predator_position", "prey_positions"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "recordings": ["behavioural choice", "position", "fish_angle", "predator_position", "prey_positions"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "recordings": ["behavioural choice", "position", "fish_angle", "predator_position", "prey_positions"],
                "interactions": []
            }
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
manager = TrialManager(current_training_configuration)
manager.run_priority_loop()
