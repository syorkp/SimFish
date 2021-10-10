import json
import os
from datetime import datetime

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

with open("./Run-Configurations/VRV_CONFIG.json", "r") as file:
    vrv_config = json.load(file)


ppo_discrete_det_training_config = [
    {
        "Model Name": "ppo_discrete_deterministic_new",
        "Environment Name": "ppo_discrete_new",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": False,
        "Continuous Actions": False,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_discrete_deterministic_new",
        "Environment Name": "ppo_discrete_new",
        "Trial Number": 2,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": False,
        "Continuous Actions": False,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_discrete_deterministic_new",
        "Environment Name": "ppo_discrete_new",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": False,
        "Continuous Actions": False,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
]


ppo_discrete_training_config = [
    {
        "Model Name": "ppo_discrete_mc",
        "Environment Name": "ppo_discrete_latest",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_discrete_mc",
        "Environment Name": "ppo_discrete_latest",
        "Trial Number": 2,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_discrete_mc",
        "Environment Name": "ppo_discrete_latest",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
]

ppo_continuous_multivariate_test_config = [
    {
        "Model Name": "ppo_continuous_multivariate",
        "Environment Name": "ppo_continuous_multivariate",
        "Trial Number": 5,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_continuous_multivariate",
        "Environment Name": "ppo_continuous_multivariate",
        "Trial Number": 6,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_continuous_multivariate",
        "Environment Name": "ppo_continuous_multivariate",
        "Trial Number": 7,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
]




ppo_continuous_training_config = [
    {
        "Model Name": "ppo_continuous_gamma_80",
        "Environment Name": "ppo_continuous_gamma_80",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
    },
    {
        "Model Name": "ppo_continuous_gamma_90",
        "Environment Name": "ppo_continuous_gamma_90",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
    },
    {
        "Model Name": "ppo_continuous_gamma_99",
        "Environment Name": "ppo_continuous_gamma_99",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
    },
]

ppo_vrv_config = [
{
        "Model Name": "ppo_multivariate_bptt",
        "Environment Name": "ppo_multivariate_assay",
        "Trial Number": 2,
        "Assay Configuration Name": "MultivariateData",
        "Total Configurations": 3,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "set random seed": True,
        "Assays": [
            {
                "assay id": "Naturalistic-1",
                "stimulus paradigm": "Presentation",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
        ]
    },
]

ppo_assay_configuration_univariate = [
    {
        "Model Name": "ppo_continuous_buffered",
        "Environment Name": "ppo_univariate_assay",
        "Trial Number": 2,
        "Assay Configuration Name": "MultivariateData",
        "Total Configurations": 3,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "set random seed": True,
        "Assays": [
            {
                "assay id": "Naturalistic-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
        ]
    },
]

ppo_assay_configuration_extra = [
#     {
#         "Model Name": "ppo_continuous_multivariate",
#         "Environment Name": "ppo_multivariate_assay",
#         "Trial Number": 7,
#         "Assay Configuration Name": "MultivariateData",
#         "Total Configurations": 3,
#         "Run Mode": "Assay",
#         "Tethered": False,
#         "Realistic Bouts": True,
#         "Continuous Actions": True,
#         "Learning Algorithm": "PPO",
#         "Priority": 2,
#         "Using GPU": True,
#         "monitor gpu": False,
#         "set random seed": True,
#         "Assays": [
#             {
#                 "assay id": "Naturalistic-5",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-6",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-7",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-8",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#         ]
#     },
#     {
#         "Model Name": "ppo_continuous_multivariate",
#         "Environment Name": "ppo_multivariate_assay",
#         "Trial Number": 9,
#         "Assay Configuration Name": "MultivariateData",
#         "Total Configurations": 3,
#         "Run Mode": "Assay",
#         "Tethered": False,
#         "Realistic Bouts": True,
#         "Continuous Actions": True,
#         "Learning Algorithm": "PPO",
#         "Priority": 2,
#         "Using GPU": True,
#         "monitor gpu": False,
#         "set random seed": True,
#         "Assays": [
#             {
#                 "assay id": "Naturalistic-5",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-6",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-7",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-8",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#         ]
#     },
# {
#         "Model Name": "ppo_multivariate_bptt",
#         "Environment Name": "ppo_multivariate_assay",
#         "Trial Number": 2,
#         "Assay Configuration Name": "MultivariateData",
#         "Total Configurations": 3,
#         "Run Mode": "Assay",
#         "Tethered": False,
#         "Realistic Bouts": True,
#         "Continuous Actions": True,
#         "Learning Algorithm": "PPO",
#         "Priority": 2,
#         "Using GPU": True,
#         "monitor gpu": False,
#         "set random seed": True,
#         "Assays": [
#             {
#                 "assay id": "Naturalistic-5",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-6",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-7",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#             {
#                 "assay id": "Naturalistic-8",
#                 "stimulus paradigm": "Naturalistic",
#                 "duration": 1000,
#                 "Tethered": False,
#                 "save frames": False,
#                 "random positions": False,
#                 "background": None,
#                 "moving": False,
#                 "save stimuli": False,
#                 "reset": False,
#                 "collisions": True,
#
#                 "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
#                 "ablations": []
#             },
#         ]
#     },
    {
        "Model Name": "ppo_continuous_buffered",
        "Environment Name": "ppo_univariate_assay",
        "Trial Number": 2,
        "Assay Configuration Name": "MultivariateData",
        "Total Configurations": 3,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "set random seed": True,
        "Assays": [
            # {
            #     "assay id": "Naturalistic-1",
            #     "stimulus paradigm": "Naturalistic",
            #     "duration": 1000,
            #     "Tethered": False,
            #     "save frames": False,
            #     "random positions": False,
            #     "background": None,
            #     "moving": False,
            #     "save stimuli": False,
            #     "reset": False,
            #     "collisions": True,
            #
            #     "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
            #     "ablations": []
            # },
            {
                "assay id": "Naturalistic-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
        ]
    },
]


ppo_assay_configuration = [
    # {
    #     "Model Name": "ppo_multivariate_bptt",
    #     "Environment Name": "ppo_multivariate_assay",
    #     "Trial Number": 2,
    #     "Assay Configuration Name": "MultivariateData",
    #     "Total Configurations": 3,
    #     "Run Mode": "Assay",
    #     "Tethered": False,
    #     "Realistic Bouts": True,
    #     "Continuous Actions": True,
    #     "Learning Algorithm": "PPO",
    #     "Priority": 2,
    #     "Using GPU": True,
    #     "monitor gpu": False,
    #     "set random seed": True,
    #     "Assays": [
    #         {
    #             "assay id": "Naturalistic-5",
    #             "stimulus paradigm": "Naturalistic",
    #             "duration": 1000,
    #             "Tethered": False,
    #             "save frames": False,
    #             "random positions": False,
    #             "background": None,
    #             "moving": False,
    #             "save stimuli": False,
    #             "reset": False,
    #             "collisions": True,
    #
    #             "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
    #             "ablations": []
    #         },
    #         {
    #             "assay id": "Naturalistic-6",
    #             "stimulus paradigm": "Naturalistic",
    #             "duration": 1000,
    #             "Tethered": False,
    #             "save frames": False,
    #             "random positions": False,
    #             "background": None,
    #             "moving": False,
    #             "save stimuli": False,
    #             "reset": False,
    #             "collisions": True,
    #
    #             "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
    #             "ablations": []
    #         },
    #         {
    #             "assay id": "Naturalistic-7",
    #             "stimulus paradigm": "Naturalistic",
    #             "duration": 1000,
    #             "Tethered": False,
    #             "save frames": False,
    #             "random positions": False,
    #             "background": None,
    #             "moving": False,
    #             "save stimuli": False,
    #             "reset": False,
    #             "collisions": True,
    #
    #             "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
    #             "ablations": []
    #         },
    #         {
    #             "assay id": "Naturalistic-8",
    #             "stimulus paradigm": "Naturalistic",
    #             "duration": 1000,
    #             "Tethered": False,
    #             "save frames": False,
    #             "random positions": False,
    #             "background": None,
    #             "moving": False,
    #             "save stimuli": False,
    #             "reset": False,
    #             "collisions": True,
    #
    #             "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
    #             "ablations": []
    #         },
    #     ]
    # },
    {
        "Model Name": "ppo_continuous_buffered",
        "Environment Name": "ppo_univariate_assay",
        "Trial Number": 2,
        "Assay Configuration Name": "MultivariateData",
        "Total Configurations": 3,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "set random seed": True,
        "Assays": [
            {
                "assay id": "Naturalistic-5",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-6",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-7",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-8",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["convolutional layers", "rnn state", "environmental positions", "reward assessments"],
                "ablations": []
            },
        ]
    },
]

sb_test_v5 = [
    {
        "Model Name": "ppo_continuous_sbv5",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone2",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True
    },
    {
        "Model Name": "ppo_continuous_sbv5",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone2",
        "Trial Number": 2,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True
    },
    {
        "Model Name": "ppo_continuous_sbv5",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone2",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True
    },
    {
        "Model Name": "ppo_continuous_sbv5",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone2",
        "Trial Number": 4,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True
    },
    ]

ppo_continuous_multivariate_sigmas_alone = [
    {
        "Model Name": "ppo_continuous_multivariate_sigmas_alone",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "ppo_continuous_multivariate_sigmas_alone",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone",
        "Trial Number": 2,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "ppo_continuous_multivariate_sigmas_alone",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "ppo_continuous_multivariate_sigmas_alone",
        "Environment Name": "ppo_continuous_multivariate_sigmas_alone",
        "Trial Number": 4,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
]


ppo_univariate_buffered = [
    {
        "Model Name": "ppo_continuous_buffered",
        "Environment Name": "ppo_continuous_buffered",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_continuous_buffered",
        "Environment Name": "ppo_continuous_buffered",
        "Trial Number": 4,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_continuous_buffered",
        "Environment Name": "ppo_continuous_buffered",
        "Trial Number": 5,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    },
    {
        "Model Name": "ppo_continuous_buffered",
        "Environment Name": "ppo_continuous_buffered",
        "Trial Number": 6,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
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
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": False,
    }
]


print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(sb_test_v5, parallel_jobs=4)
manager.run_priority_loop()
