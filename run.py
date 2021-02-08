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
        "Environment Name": "naturalistic",
        "Assay Configuration Name": "Naturalistic",
        "Trial Number": 2,
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
            # {
            #     "assay id": "Naturalistic-2",
            #     "stimulus paradigm": "Naturalistic",
            #     "duration": 1000,
            #     "Tethered": False,
            #     "save frames": False,
            #     "recordings": ["behavioural choice", "position", "fish_angle", "predator_position", "prey_positions"],
            #     "interactions": []
            # },
            # {
            #     "assay id": "Naturalistic-3",
            #     "stimulus paradigm": "Naturalistic",
            #     "duration": 1000,
            #     "Tethered": False,
            #     "save frames": False,
            #     "recordings": ["behavioural choice", "position", "fish_angle", "predator_position", "prey_positions"],
            #     "interactions": []
            # },
            # {
            #     "assay id": "Naturalistic-4",
            #     "stimulus paradigm": "Naturalistic",
            #     "duration": 1000,
            #     "Tethered": False,
            #     "save frames": False,
            #     "recordings": ["behavioural choice", "position", "fish_angle", "predator_position", "prey_positions"],
            #     "interactions": []
            # }
        ],
    },
]


current_training_configuration = [
    {
        "Model Name": "changed_penalties",
        "Environment Name": "changed_penalties_1",
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "changed_penalties",
        "Environment Name": "changed_penalties_2",
        "Trial Number": 2,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "larger_network",
        "Environment Name": "larger_network",
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "smaller_network",
        "Environment Name": "smaller_network",
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(current_assay_configuration)
manager.run_priority_loop()
