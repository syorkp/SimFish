import os
from datetime import datetime

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")


prey_assay_config = [
    {
        "Model Name": "changed_penalties",
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
                "save frames": True,
                "random positions": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
        ],
    },
]


predator_assay_config = [
    {
        "Model Name": "changed_penalties",
        "Environment Name": "predator_heavy",
        "Assay Configuration Name": "Predator",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Assays": [
            {
                "assay id": "Predator-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
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


non_random_projection_configuration = [
    {
        "Model Name": "changed_penalties",
        "Environment Name": "test_square",
        "Trial Number": 1,
        "Assay Configuration Name": "Controlled_Visual_Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Curved_prey",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "reset": True,
                "reset interval": 100,
                "duration": 500,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "prey 1": {"steps": 500,
                               "size": 5,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
        ],
    }
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(non_random_projection_configuration)
manager.run_priority_loop()

