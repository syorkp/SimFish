import os
from datetime import datetime

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")


current_training_configuration = [
    {
        "Model Name": "new_actions_all_features",
        "Environment Name": "new_actions_all_features",
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 22,
                "4": 27,
                "5": 32,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 20,
            }
        },
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": True,
        "Fish Setup": "Free",
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "new_actions_all_features",
        "Environment Name": "new_actions_all_features",
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 22,
                "4": 27,
                "5": 32,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 20,
            }
        },
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": True,
        "Fish Setup": "Free",
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "new_actions_all_features",
        "Environment Name": "new_actions_all_features",
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 22,
                "4": 27,
                "5": 32,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 20,
            }
        },
        "Trial Number": 3,
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(current_training_configuration)
manager.run_priority_loop()
