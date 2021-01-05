import os

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

# TODO: Move whole thing to its own JSON configuration set.
controlled_assay_configuration = [
    {
        "Model Name": "base",
        "Environment Name": "test_square",
        "Trial Number": 1,
        "Assay Configuration Name": "Prey Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Visual-Stimulus-Assay-2",
                "stimulus paradigm": "Projection",
                "duration": 121,
                "fish setup": "Tethered",
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "prey 1": [
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
                    ],
                },
                "interactions": []},
            # {
            #     "assay id": "Visual-Stimulus-Assay-2",
            #     "stimulus paradigm": "Projection",
            #     "duration": 300,
            #     "fish setup": "Tethered",
            #     "save frames": True,
            #     "recordings": ["behavioural choice", "rnn state", "observation"],
            #     "stimuli": {
            #         "predator 1": [
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
            #         ]
            #     }
            # }
        ]
    },
]

naturalistic_assay_configuration = [
    {
            "Model Name": "base",
            "Environment Name": "base",
            "Assay Configuration Name": "Naturalistic",
            "Trial Number": 1,
            "Run Mode": "Assay",
            "Priority": 1,
            "Realistic Bouts": False,
            "Assays": [
                    {
                        "assay id": "Naturalistic-Assay-1",
                        "stimulus paradigm": "Naturalistic",
                        "duration": 400,
                        "fish setup": "Free",
                        "save frames": True,
                        "recordings": [],
                        "interactions": []
                    }
                ],
    }
]

training_configuration = [
    {
        "Model Name": "scaffolding_test",
        "Environment Name": "gradual",
        "Total Configurations": 8,
        "Episode Transitions": {
            "2": 400,
            "3": 800,
            "4": 1200,
            "5": 1600,
            "6": 1800,
            "7": 2000,
            "8": 2200,
        },
        "Conditional Transitions": {
            "Prey Caught": {
            },
            "Predators Avoided": {
            }
        },
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": False,
        "Priority": 1,
        "monitor gpu": False,
    },
]

# TODO: Change fish steup to tethered boolean.

manager = TrialManager(training_configuration)
manager.run_priority_loop()
