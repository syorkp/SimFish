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
                "to record": ["advantage stream", "behavioural choice", "rnn state", "position"]
            }
        ]
    },
]

trial_configuration = [
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
                "to record": ["advantage stream", "behavioural choice", "rnn state", "position", "observation"]
            }
        ]
    },
]

training_configuration = [
    # {
    #     "Environment Name": "base",
    #     "Trial Number": 2,
    #     "Run Mode": "Training",
    #     "Fish Setup": "Free",
    #     "Priority": 1,
    # },
    # {
    #     "Environment Name": "new-test",
    #     "Trial Number": 1,
    #     "Run Mode": "Training",
    #     "Fish Setup": "Free",
    #     "Priority": 2,
    # },
    {
        "Environment Name": "base",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Fish Setup": "Free",
        "Priority": 3,
        "Assays": [
            {
                "assay id": "Assay-2",
                "stimulus": "Normal environment",
                "end requirement": "Death or 1000 steps",
                "save frames": "True",
                "to record": ["advantage stream", "behavioural choice", "rnn state", "position", "observation"]
            },
            {
                "assay id": "Assay-3",
                "stimulus": "Normal environment",
                "end requirement": "Death or 1000 steps",
                "save frames": "True",
                "to record": ["advantage stream", "behavioural choice", "rnn state", "position", "observation"]
            },
        ]
    },
]

# Run the configuration creator
myriad_job = [
    {
        "Environment Name": "base",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Priority": 1,
    },
]

manager = TrialManager(myriad_job)
manager.run_priority_loop()
