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
        "Model Name": "continuous_extra_rnn_learning",
        "Environment Name": "continuous_extra_rnn_learning",
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
        "Continuous Actions": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    }
]


print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(test_continuous_training_configuration)
manager.run_priority_loop()
