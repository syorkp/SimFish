import os

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

trial_configuration_example = [
    {
        "Environment Name": "base",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Priority": 1
    },
    {
        "Environment Name": "base",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Priority": 2
    },
    {
        "Environment Name": "base",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Priority": 1
    },
    {
        "Environment Name": "test",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Priority": 3
    },
    {
        "Environment Name": "new_test",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Fish Setup": "Free",
        "Priority": 3
    },
]

trial_configuration = [
    {
        "Environment Name": "base",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Fish Setup": "Free",
        "Priority": 1
    },
]

# Run the configuration creator


manager = TrialManager(trial_configuration)
manager.run_priority_loop()
