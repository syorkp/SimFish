import os

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

trial_configuration_examples = [
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
        "Environment Name": "base",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Fish Setup": "Free",
        "Priority": 3,
        "Assays": [
            {
                "assay id": "Assay-1",
                "stimulus": "Normal environment",
                "end requirement": "Death or 1000 steps",
                "to record": ["advantage stream", "behavioural choice", "rnn state"]
            }
        ]
    },
]

trial_configuration = [
    {
        "Environment Name": "base",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Assay-1",
                "stimulus": "Normal environment",
                "end requirement": "Death or 1000 steps",
                "to record": ["advantage stream", "behavioural choice", "rnn state"]
            }
        ]

    },
]

# Run the configuration creator


manager = TrialManager(trial_configuration)
manager.run_priority_loop()
