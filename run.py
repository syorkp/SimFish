import os
from datetime import datetime
import json

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")


test_continuous_training_configuration = [
    {
        "Model Name": "scaffold_test_high_learning_rate",
        "Environment Name": "continuous_learning_scaffold",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    }
]


test_continuous_assay_configuration = [
    {
        "Model Name": "scaffold_test",
        "Environment Name": "continuous_assay",
        "Trial Number": 1,
        "Assay Configuration Name": "Continuous_Action_Mapping",
        "Total Configurations": 3,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "set random seed": False,
        "Assays": [
            {
                "assay id": "Environment-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["impulse", "angle", "position", "fish_angle", "predator", "prey_positions",
                               "rnn state"],
                "ablations": []
            },
        ]
    }
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(test_continuous_training_configuration)
manager.run_priority_loop()
