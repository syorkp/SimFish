import os
from datetime import datetime

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

test_config = [
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "naturalistic",
        "Trial Number": 5,
        "Assay Configuration Name": "Ablation-Test",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Naturalistic-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 500,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": [],
                "ablations": [i for i in range(500)]
            },
            ]
    }
    ]



current_training_configuration = [
    {
        "Model Name": "no_predators_ref",
        "Environment Name": "no_predators",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
                "4": 20,
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
        "Model Name": "even_prey_ref",
        "Environment Name": "even_prey",
        "Trial Number": 5,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
                "4": 20,
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

full_response_vector_config = [
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "test_square",
        "Trial Number": 5,
        "Assay Configuration Name": "Predator-Full-Response-Vector",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Predator-Static-40",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": False,
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 40,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Static-60",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": False,
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 60,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Static-80",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": False,
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 80,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Left-40",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Left",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 40,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Left-60",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Left",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 60,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Left-80",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Left",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 80,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Right-40",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Right",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 40,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Right-60",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Right",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 60,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Right-80",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Right",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 80,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Away",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Away",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 80,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Predator-Towards",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Towards",
                "reset": False,
                "background": None,
                "reset interval": 100,
                "duration": 1100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 1100,
                               "size": 40,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
        ]
    },
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(test_config)
manager.run_priority_loop()
