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
prey_only = [13, 21, 29, 44, 51, 52, 53, 57, 59, 60, 65, 69, 76, 86, 90, 115, 116, 121, 122, 129, 138, 142, 149, 157, 171, 173, 175, 176, 182, 183, 186, 188, 191, 201, 203, 205, 221, 225, 232, 239, 250, 260, 268, 292, 303, 312, 321, 327, 328, 347, 354, 362, 366, 368, 390, 395, 399, 402, 403, 406, 415, 429, 446, 447, 456, 463, 481, 497, 504]
pred_only = [4, 5, 9, 10, 15, 16, 17, 22, 28, 34, 41, 42, 46, 47, 49, 55, 56, 64, 70, 77, 79, 85, 89, 93, 95, 98, 99, 100, 101, 106, 114, 118, 120, 133, 134, 135, 136, 139, 140, 145, 156, 158, 163, 166, 172, 174, 178, 179, 181, 189, 193, 194, 198, 200, 204, 206, 208, 213, 220, 224, 234, 236, 245, 249, 251, 253, 255, 257, 261, 263, 266, 269, 271, 275, 295, 296, 307, 317, 329, 338, 345, 346, 352, 364, 372, 381, 383, 385, 388, 418, 421, 424, 436, 449, 450, 453, 454, 458, 462, 468, 470, 482, 483, 489, 492, 495, 501, 503, 505, 509, 511]
import random
uv = [12, 14, 35, 72, 252, 359, 360, 363, 384, 498]


test_config = [
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "naturalistic",
        "Trial Number": 5,
        "Assay Configuration Name": "Ablation-Test-Unvexed",
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
                "ablations": uv,
            },
            ]
    },
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "naturalistic",
        "Trial Number": 5,
        "Assay Configuration Name": "Ablation-Test-Unvexed-Random",
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
                "ablations": list(random.sample(range(0, 512), len(uv))),
            },
        ]
    }
    ]

predator_heavy_config = [
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "predator_heavy",
        "Assay Configuration Name": "TP-Calculation",
        "Trial Number": 5,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "set random seed": False,
        "Assays": [
            {
                "assay id": "Prey-Predator-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-5",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-6",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-7",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-8",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-9",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-10",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
{
                "assay id": "Prey-Predator-11",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-12",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-13",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-14",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-15",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-16",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-17",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-18",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-19",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
            },
            {
                "assay id": "Prey-Predator-20",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "ablations": []
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
manager = TrialManager(predator_heavy_config)
manager.run_priority_loop()
