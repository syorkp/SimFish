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
        "Model Name": "scaffold_test",
        "Environment Name": "continuous_learning_scaffold",
        "Trial Number": 15,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
        "Trial Number": 12,
        "Assay Configuration Name": "Checking_Observation",
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
                "duration": 200,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["left_conv_4", "rnn state", "consumed"],
                "ablations": []
            },
        ]
    }
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(test_continuous_training_configuration)
manager.run_priority_loop()
