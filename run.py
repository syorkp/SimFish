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
        "Model Name": "large_all_features",
        "Environment Name": "naturalistic",
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
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-5",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-6",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-7",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-8",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-9",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-10",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-11",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
            {
                "assay id": "Naturalistic-12",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice"],
                "interactions": []
            },
        ],
    },
]


test_training_configuration = [
    {
        "Model Name": "test",
        "Environment Name": "ref_test",
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
]

current_training_configuration = [
    {
        "Model Name": "differential_prey_ref",
        "Environment Name": "differential_prey",
        "Trial Number": 2,
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
        "Trial Number": 2,
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


predator_projection = [
    {
        "Model Name": "large_all_features",
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
                "assay id": "Predator-Static",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Left",
                "reset": False,
                "reset interval": 100,
                "duration": 500,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rnn state"],
                "stimuli": {
                    "predator 1": {"steps": 500,
                                   "size": 50,
                                   "interval": 100,
                                   },
                },
                "interactions": []
            },
        ]
    }
]

non_random_projection_configuration = [
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "test_square_2",
        "Trial Number": 1,
        "Assay Configuration Name": "Controlled_Visual_Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Prey-Moving-Left",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": "Left",
                "reset": False,
                "reset interval": 100,
                "duration": 100,
                "save frames": True,
                "save stimuli": True,
                "recordings": ["rev_observation", "observation"],
                "stimuli": {
                    "prey 1": {"steps": 100,
                               "size": 5,
                               "interval": 50,
                               },
                },
                "interactions": []
            },
            # {
            #     "assay id": "Prey-Moving-Right",
            #     "stimulus paradigm": "Projection",
            #     "Tethered": True,
            #     "set positions": False,
            #     "random positions": False,
            #     "moving": "Right",
            #     "reset": False,
            #     "reset interval": 100,
            #     "duration": 1000,
            #     "save frames": True,
            #     "save stimuli": True,
            #     "recordings": ["rnn state"],
            #     "stimuli": {
            #         "prey 1": {"steps": 1000,
            #                    "size": 5,
            #                    "interval": 100,
            #                    },
            #     },
            #     "interactions": []
            # },
            # {
            #     "assay id": "Prey-Static",
            #     "stimulus paradigm": "Projection",
            #     "Tethered": True,
            #     "set positions": False,
            #     "random positions": False,
            #     "moving": False,
            #     "reset": False,
            #     "reset interval": 100,
            #     "duration": 1000,
            #     "save frames": True,
            #     "save stimuli": True,
            #     "recordings": ["rnn state"],
            #     "stimuli": {
            #         "prey 1": {"steps": 1000,
            #                    "size": 5,
            #                    "interval": 100,
            #                    },
            #     },
            #     "interactions": []
            # },
        ],
    }
]

no_stimuli_projection_config = [
    {
        "Model Name": "large_all_features",
        "Environment Name": "test_square",
        "Trial Number": 1,
        "Assay Configuration Name": "No_Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "No_Stimuli",
                "stimulus paradigm": "Projection",
                "Tethered": True,
                "set positions": False,
                "random positions": False,
                "moving": False,
                "reset": False,
                "reset interval": 500,
                "duration": 10000,
                "save frames": False,
                "save stimuli": False,
                "recordings": ["rnn state", "behavioural choice"],
                "stimuli": {
                },
                "interactions": []
            },
        ],
    }
]


print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(current_training_configuration)
manager.run_priority_loop()

