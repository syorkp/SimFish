import os
from datetime import datetime

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

prey_15 = [0, 6, 11, 21, 25, 26, 27, 29, 31, 32, 44, 71, 73, 74, 82, 87, 104, 107, 109, 113, 117, 121, 128, 141, 143, 144, 155, 160, 184, 192, 203, 205, 207, 209, 215, 227, 240, 242, 246, 254, 268, 270, 277, 287, 297, 298, 303, 308, 310, 311, 315, 319, 322, 323, 325, 328, 331, 333, 334, 341, 343, 344, 361, 362, 366, 376, 396, 397, 400, 403, 408, 412, 413, 416, 419, 425, 426, 427, 428, 429, 430, 433, 435, 437, 439, 440, 442, 446, 447, 463, 464, 466, 472, 476, 477, 481, 485, 486, 487, 488, 491, 496, 499, 507, 510]

random_ns = [11, 23, 29, 34, 40, 41, 69, 80, 81, 88, 100, 101, 109, 112, 117, 124, 130, 162, 169, 170, 171, 176, 181, 189, 190, 191, 197, 201, 206, 207, 213, 215, 216, 221, 222, 223, 230, 231, 234, 242, 248, 249, 252, 258, 259, 261, 263, 265, 268, 273, 278, 284, 286, 287, 294, 295, 296, 298, 301, 304, 307, 308, 315, 328, 341, 342, 348, 349, 350, 352, 359, 360, 363, 371, 377, 381, 383, 385, 387, 389, 390, 392, 409, 417, 423, 429, 430, 431, 434, 436, 437, 448, 461, 465, 474, 475, 482, 490, 491, 493, 495, 498, 500, 505, 508]
random_ns2 = [9, 10, 13, 24, 32, 34, 36, 41, 44, 50, 52, 58, 66, 68, 69, 77, 81, 86, 88, 91, 94, 101, 102, 107, 117, 125, 128, 144, 151, 155, 156, 161, 166, 172, 181, 183, 189, 191, 192, 193, 194, 197, 201, 203, 205, 211, 234, 238, 241, 254, 260, 273, 288, 290, 293, 297, 299, 300, 301, 305, 311, 318, 321, 322, 330, 342, 352, 355, 356, 365, 369, 373, 378, 380, 391, 399, 402, 403, 405, 408, 409, 411, 414, 415, 416, 420, 422, 432, 434, 437, 440, 450, 460, 464, 468, 471, 472, 473, 476, 483, 484, 485, 486, 503, 506]


test_config = [
    # {
    #     "Model Name": "even_prey_ref",
    #     "Environment Name": "naturalistic",
    #     "Trial Number": 5,
    #     "Assay Configuration Name": "Ablation-Test-1",
    #     "Run Mode": "Assay",
    #     "Realistic Bouts": True,
    #     "Using GPU": False,
    #     "monitor gpu": False,
    #     "Priority": 1,
    #     "set random seed": True,
    #     "Assays": [
    #         {
    #             "assay id": "Naturalistic-1",
    #             "stimulus paradigm": "Naturalistic",
    #             "duration": 500,
    #             "Tethered": False,
    #             "save frames": True,
    #             "random positions": False,
    #             "background": None,
    #             "moving": False,
    #             "save stimuli": False,
    #             "reset": False,
    #             "recordings": ["consumed"],
    #             "ablations": [],
    #         },
    #         ]
    # },
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "naturalistic",
        "Trial Number": 5,
        "Assay Configuration Name": "Ablation-Test-4",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Priority": 1,
        "set random seed": True,
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
                "recordings": ["consumed"],
                "ablations": random_ns2,
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
