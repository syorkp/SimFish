import os

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

# TODO: Move whole thing to its own JSON configuration set.
assay_configuration = [
    {
        "Model Name": "base",
        "Environment Name": "test_square",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Assays": [
            # {
            #     "assay id": "Visual-Stimulus-Assay-1",
            #     "stimulus paradigm": "Projection",
            #     "duration": 300,
            #     "fish setup": "Tethered",
            #     "save frames": True,
            #     "recordings": ["behavioural choice", "rnn state", "observation"],
            #     "stimuli": {
            #         "prey 1": [
            #             {"step": 0,
            #              "position": [100, 100]},
            #             {"step": 20,
            #              "position": [300, 100]},
            #             {"step": 40,
            #              "position": [300, 300]},
            #             {"step": 60,
            #              "position": [100, 300]},
            #             {"step": 80,
            #              "position": [300, 300]},
            #             {"step": 100,
            #              "position": [300, 100]},
            #             {"step": 120,
            #              "position": [100, 100]},
            #         ],
            #     },
            #     "interactions": []},
            {
                "assay id": "Visual-Stimulus-Assay-2",
                "stimulus paradigm": "Projection",
                "duration": 300,
                "fish setup": "Tethered",
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "predator 1": [
                        {"step": 0,
                         "position": [100, 100]},
                        {"step": 20,
                         "position": [300, 100]},
                        {"step": 40,
                         "position": [300, 300]},
                        {"step": 60,
                         "position": [100, 300]},
                        {"step": 80,
                         "position": [300, 300]},
                        {"step": 100,
                         "position": [300, 100]},
                        {"step": 120,
                         "position": [100, 100]},
                    ]
                }
            }
        ]
    },
]

training_configuration = [
    {
        "Model Name": "base",
        "Environment Name": "base",
        "Trial Number": 6,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Priority": 1,
        "using gpu": False,
        "monitor gpu": True,
    },
]

manager = TrialManager(assay_configuration)
manager.run_priority_loop()
