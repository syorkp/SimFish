import os
from datetime import datetime

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
        "Using GPU": True,
        "monitor gpu": True,
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
            "Using GPU": True,
            "monitor gpu": True,
            "Assays": [
                    {
                        "assay id": "Vegetation-Effects",
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

asaph_data_configuration = [
    # {
    #     "Model Name": "realistic_action_space",
    #     "Environment Name": "example",
    #     "Assay Configuration Name": "realistic_actions",
    #     "Trial Number": 1,
    #     "Run Mode": "Assay",
    #     "Priority": 1,
    #     "Realistic Bouts": True,
    #     "Using GPU": False,
    #     "monitor gpu": False,
    #     "Assays": [
    #         {
    #             "assay id": "All-Features",
    #             "stimulus paradigm": "Naturalistic",
    #             "duration": 1000,
    #             "fish setup": "Free",
    #             "save frames": True,
    #             "recordings": ["behavioural choice", "rnn state", "observation", "position"],
    #             "interactions": []
    #         }
    #     ],
    # },
    {
        "Model Name": "conditional_prey_and_predators",
        "Environment Name": "example",
        "Assay Configuration Name": "simple_actions",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": False,
        "Using GPU": False,
        "monitor gpu": False,
        "Assays": [
            {
                "assay id": "All-Features",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "fish setup": "Free",
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation", "position"],
                "interactions": []
            }
        ],
    },
    # {
    #     "Model Name": "realistic_action_space",
    #     "Environment Name": "test_square",
    #     "Trial Number": 1,
    #     "Assay Configuration Name": "Prey Stimuli",
    #     "Run Mode": "Assay",
    #     "Realistic Bouts": True,
    #     "Using GPU": False,
    #     "monitor gpu": False,
    #     "Priority": 1,
    #     "Assays": [
    #         {
    #             "assay id": "Visual-Stimulus-Assay-2",
    #             "stimulus paradigm": "Projection",
    #             "duration": 121,
    #             "fish setup": "Tethered",
    #             "save frames": True,
    #             "recordings": ["behavioural choice", "rnn state", "observation", "position"],
    #             "stimuli": {
    #                 "prey 1": [
    #                     {"step": 0,
    #                      "position": [100, 100]},
    #                     {"step": 20,
    #                      "position": [300, 100]},
    #                     {"step": 40,
    #                      "position": [300, 300]},
    #                     {"step": 60,
    #                      "position": [100, 300]},
    #                     {"step": 80,
    #                      "position": [300, 300]},
    #                     {"step": 100,
    #                      "position": [300, 100]},
    #                     {"step": 120,
    #                      "position": [100, 100]},
    #                 ],
    #             },
    #             "interactions": []
    #         },
    #     ],
    # }
]

training_configuration = [
    {
        "Model Name": "reduce_mouth",
        "Environment Name": "increasing_prey_speed_3",
        "Total Configurations": 7,
        "Episode Transitions": {
            "2": 400,
            "3": 800,
            "4": 1000,
            "5": 1200,
            "6": 1400,
            "7": 1800,
        },
        "Conditional Transitions": {
            "Prey Caught": {
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": False,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": True,
    },
    {
        "Model Name": "modified_action_costs",
        "Environment Name": "increasing_prey_speed_4",
        "Total Configurations": 5,
        "Episode Transitions": {
            "2": 800,
            "3": 1200,
            "4": 1600,
            "5": 2000,
        },
        "Conditional Transitions": {
            "Prey Caught": {
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": False,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": True,
    },
    {
        "Model Name": "earlier_transition",
        "Environment Name": "increasing_prey_speed_1",
        "Total Configurations": 5,
        "Episode Transitions": {
            "2": 400,
            "3": 600,
            "4": 800,
            "5": 1000,
        },
        "Conditional Transitions": {
            "Prey Caught": {
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Trial Number": 2,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": False,
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": True,
    },
    {
        "Model Name": "conditional_transfer",
        "Environment Name": "increasing_prey_speed_1",
        "Total Configurations": 6,
        "Episode Transitions": {

        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 20,
                "3": 25,
                "4": 27,
                "5": 29,
                "6": 31,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": False,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": True,
    },
]

new_training_configuration = [
    {
        "Model Name": "realistic_all_features",
        "Environment Name": "all_features",
        "Total Configurations": 9,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "5": 22,
                "6": 27,
                "7": 32,
            },
            "Predators Avoided": {
                "2": 4,
                "3": 6,
                "4": 8,
            },
            "Sand Grains Bumped": {
                "8": 20,
                "9": 10,
            }
        },
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "realistic_all_features",
        "Environment Name": "all_features",
        "Total Configurations": 9,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "5": 20,
                "6": 24,
                "7": 28,
            },
            "Predators Avoided": {
                "2": 2,
                "3": 4,
                "4": 6,
            },
            "Sand Grains Bumped": {
                "8": 30,
                "9": 20,
            }
        },
        "Trial Number": 2,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "basic_all_features",
        "Environment Name": "all_features",
        "Total Configurations": 9,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "5": 22,
                "6": 27,
                "7": 32,
            },
            "Predators Avoided": {
                "2": 4,
                "3": 6,
                "4": 8,
            },
            "Sand Grains Bumped": {
                "8": 20,
                "9": 10,
            }
        },
        "Trial Number": 1,
        "Run Mode": "Training",
        "Fish Setup": "Free",
        "Realistic Bouts": False,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
]

# TODO: Change fish steup to tethered boolean.
print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(new_training_configuration)
manager.run_priority_loop()
