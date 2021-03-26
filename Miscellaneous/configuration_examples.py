
controlled_assay_configuration_2 = [
    {
        "Model Name": "basic_all_features",
        "Environment Name": "test_square",
        "Trial Number": 1,
        "Assay Configuration Name": "Controlled_Visual_Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": False,
        "Using GPU": True,
        "monitor gpu": True,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Moving-Prey",
                "stimulus paradigm": "Projection",
                "duration": 241,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "prey 1": [
                        {"step": 0,
                         "position": [150, 150]},
                        {"step": 40,
                         "position": [450, 150]},
                        {"step": 80,
                         "position": [450, 450]},
                        {"step": 120,
                         "position": [150, 450]},
                        {"step": 160,
                         "position": [450, 450]},
                        {"step": 200,
                         "position": [450, 150]},
                        {"step": 240,
                         "position": [150, 150]},
                    ],
                },
                "interactions": []},
{
                "assay id": "Moving-Predator",
                "stimulus paradigm": "Projection",
                "duration": 121,
                "fish setup": "Tethered",
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "predator 1": [
                        {"step": 0,
                         "position": [100, 100]},
                        {"step": 20,
                         "position": [500, 100]},
                        {"step": 40,
                         "position": [500, 500]},
                        {"step": 60,
                         "position": [100, 500]},
                        {"step": 80,
                         "position": [500, 500]},
                        {"step": 100,
                         "position": [500, 100]},
                        {"step": 120,
                         "position": [100, 100]},
                    ],
                },
                "interactions": []},
        ]
    },
    {
        "Model Name": "realistic_all_features",
        "Environment Name": "test_square",
        "Trial Number": 2,
        "Assay Configuration Name": "Controlled_Visual_Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": True,
        "monitor gpu": True,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Moving-Prey",
                "stimulus paradigm": "Projection",
                "duration": 241,
                "fish setup": "Tethered",
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "prey 1": [
                        {"step": 0,
                         "position": [150, 150]},
                        {"step": 40,
                         "position": [450, 150]},
                        {"step": 80,
                         "position": [450, 450]},
                        {"step": 120,
                         "position": [150, 450]},
                        {"step": 160,
                         "position": [450, 450]},
                        {"step": 200,
                         "position": [450, 150]},
                        {"step": 240,
                         "position": [150, 150]},
                    ],
                },
                "interactions": []
            },
            {
                "assay id": "Moving-Predator",
                "stimulus paradigm": "Projection",
                "duration": 121,
                "fish setup": "Tethered",
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "predator 1": [
                        {"step": 0,
                         "position": [100, 100]},
                        {"step": 20,
                         "position": [500, 100]},
                        {"step": 40,
                         "position": [500, 500]},
                        {"step": 60,
                         "position": [100, 500]},
                        {"step": 80,
                         "position": [500, 500]},
                        {"step": 100,
                         "position": [500, 100]},
                        {"step": 120,
                         "position": [100, 100]},
                    ],
                },
                "interactions": []},
        ]
    },
]


controlled_assay_configuration_2 = [
    {
        "Model Name": "basic_all_features",
        "Environment Name": "test_square",
        "Trial Number": 1,
        "Assay Configuration Name": "Controlled_Visual_Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": False,
        "Using GPU": True,
        "monitor gpu": True,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Moving-Prey",
                "stimulus paradigm": "Projection",
                "duration": 241,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "prey 1": [
                        {"step": 0,
                         "position": [150, 150]},
                        {"step": 40,
                         "position": [450, 150]},
                        {"step": 80,
                         "position": [450, 450]},
                        {"step": 120,
                         "position": [150, 450]},
                        {"step": 160,
                         "position": [450, 450]},
                        {"step": 200,
                         "position": [450, 150]},
                        {"step": 240,
                         "position": [150, 150]},
                    ],
                },
                "interactions": []},
{
                "assay id": "Moving-Predator",
                "stimulus paradigm": "Projection",
                "duration": 121,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "predator 1": [
                        {"step": 0,
                         "position": [100, 100]},
                        {"step": 20,
                         "position": [500, 100]},
                        {"step": 40,
                         "position": [500, 500]},
                        {"step": 60,
                         "position": [100, 500]},
                        {"step": 80,
                         "position": [500, 500]},
                        {"step": 100,
                         "position": [500, 100]},
                        {"step": 120,
                         "position": [100, 100]},
                    ],
                },
                "interactions": []},
        ]
    },
    {
        "Model Name": "realistic_all_features",
        "Environment Name": "test_square",
        "Trial Number": 2,
        "Assay Configuration Name": "Controlled_Visual_Stimuli",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": True,
        "monitor gpu": True,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Moving-Prey",
                "stimulus paradigm": "Projection",
                "duration": 241,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "prey 1": [
                        {"step": 0,
                         "position": [150, 150]},
                        {"step": 40,
                         "position": [450, 150]},
                        {"step": 80,
                         "position": [450, 450]},
                        {"step": 120,
                         "position": [150, 450]},
                        {"step": 160,
                         "position": [450, 450]},
                        {"step": 200,
                         "position": [450, 150]},
                        {"step": 240,
                         "position": [150, 150]},
                    ],
                },
                "interactions": []
            },
            {
                "assay id": "Moving-Predator",
                "stimulus paradigm": "Projection",
                "duration": 121,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "predator 1": [
                        {"step": 0,
                         "position": [100, 100]},
                        {"step": 20,
                         "position": [500, 100]},
                        {"step": 40,
                         "position": [500, 500]},
                        {"step": 60,
                         "position": [100, 500]},
                        {"step": 80,
                         "position": [500, 500]},
                        {"step": 100,
                         "position": [500, 100]},
                        {"step": 120,
                         "position": [100, 100]},
                    ],
                },
                "interactions": []},
        ]
    },
]

predators_only_config = [
    {
        "Model Name": "basic_all_features",
        "Environment Name": "predators_only",
        "Assay Configuration Name": "Predator-Only-Avoidance",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": False,
        "Using GPU": False,
        "monitor gpu": False,
        "Assays": [
            {
                "assay id": "Predator_Avoidance-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator_Avoidance-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator_Avoidance-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator_Avoidance-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
        ]
    }
]

behaviour_sequence_data_configuration_basic = [
    {
        "Model Name": "basic_all_features",
        "Environment Name": "prey",
        "Assay Configuration Name": "Prey-Capture",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": False,
        "Using GPU": False,
        "monitor gpu": False,
        "Assays": [
            {
                "assay id": "Prey_Capture-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Prey_Capture-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Prey_Capture-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Prey_Capture-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            }
        ],
    },
    {
        "Model Name": "basic_all_features",
        "Environment Name": "predators",
        "Assay Configuration Name": "Predator-Avoidance",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": False,
        "Using GPU": False,
        "monitor gpu": False,
        "Assays": [
            {
                "assay id": "Predator_Avoidance-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator_Avoidance-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator_Avoidance-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
            {
                "assay id": "Predator_Avoidance-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "consumed", "predator"],
                "interactions": []
            },
        ]
    }
]



various_training_configuration = [
    {
        "Model Name": "changed_penalties",
        "Environment Name": "changed_penalties_1",
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "changed_penalties",
        "Environment Name": "changed_penalties_2",
        "Trial Number": 2,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "larger_network",
        "Environment Name": "larger_network",
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
    {
        "Model Name": "smaller_network",
        "Environment Name": "smaller_network",
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "3": 3,
                "4": 4,
                "5": 5,
            },
            "Predators Avoided": {
                "2": 2,
            },
            "Sand Grains Bumped": {
                "6": 10,
            }
        },
        "Run Mode": "Training",
        "Tethered": True,
        "Realistic Bouts": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    },
]

test_assay_configuration = [
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
        "Model Name": "realistic_all_features",
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
                "duration": 10,
                "Tethered": True,
                "save frames": True,
                "recordings": [#"behavioural choice", "rnn state", "observation", "position", "consumed", "predator",
                               #"left_conv_1", "left_conv_2", "left_conv_3", "left_conv_4",
                               #"right_conv_1", "right_conv_2", "right_conv_3", "right_conv_4",
                               "prey_positions", "predator_position", "sand_grain_positions", "vegetation_positions", "fish_angle"
                              ],
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
                "Tethered": True,
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
            "Model Name": "realistic_all_features",
            "Environment Name": "base",
            "Assay Configuration Name": "Naturalistic",
            "Trial Number": 2,
            "Run Mode": "Assay",
            "Priority": 1,
            "Realistic Bouts": True,
            "Using GPU": True,
            "monitor gpu": True,
            "Assays": [
                    {
                        "assay id": "All-Features-1",
                        "stimulus paradigm": "Naturalistic",
                        "duration": 1000,
                        "fish setup": "Free",
                        "save frames": True,
                        "recordings": [],
                        "interactions": []
                    },
                {
                    "assay id": "All-Features-2",
                    "stimulus paradigm": "Naturalistic",
                    "duration": 1000,
                    "fish setup": "Free",
                    "save frames": True,
                    "recordings": [],
                    "interactions": []
                },
                {
                        "assay id": "All-Features-3",
                        "stimulus paradigm": "Naturalistic",
                        "duration": 1000,
                        "fish setup": "Free",
                        "save frames": True,
                        "recordings": [],
                        "interactions": []
                    }
                ],
    },
    {
        "Model Name": "basic_all_features",
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
                "assay id": "All-Features-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "fish setup": "Free",
                "save frames": True,
                "recordings": [],
                "interactions": []
            },
            {
                "assay id": "All-Features-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "fish setup": "Free",
                "save frames": True,
                "recordings": [],
                "interactions": []
            },
            {
                "assay id": "All-Features-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "fish setup": "Free",
                "save frames": True,
                "recordings": [],
                "interactions": []
            }
        ],
    },
]

random_projection_configuration = [
    {
        "Model Name": "changed_penalties",
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
                "assay id": "Random-Prey",
                "stimulus paradigm": "Projection",
                "random positions": True,
                "reset": False,
                "duration": 241,
                "Tethered": True,
                "save frames": True,
                "recordings": ["behavioural choice", "rnn state", "observation"],
                "stimuli": {
                    "prey 1": {"steps": 240,
                               "size": 10
                               },
                },
                "interactions": []
            }
        ],
    }
]

predator_assay_config = [
    {
        "Model Name": "even_prey",
        "Environment Name": "predator_heavy",
        "Assay Configuration Name": "Predator",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Priority": 1,
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Assays": [
            {
                "assay id": "Predator-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": False,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position", "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-2",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position", "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-3",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position", "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-4",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position", "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-5",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-6",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-7",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-8",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-9",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-10",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-11",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
            {
                "assay id": "Predator-12",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save stimuli": False,
                "save frames": False,
                "random positions": False,
                "reset": False,
                "recordings": ["position", "behavioural choice", "fish_angle", "prey_positions", "predator_position",
                               "observation"],
                "interactions": []
            },
        ],
    },
]


# FULL RESPONSE VECTOR FOR PREY

full_response_vector_config = [
    {
        "Model Name": "even_prey_ref",
        "Environment Name": "test_square",
        "Trial Number": 5,
        "Assay Configuration Name": "Prey-Full-Response-Vector",
        "Run Mode": "Assay",
        "Realistic Bouts": True,
        "Using GPU": False,
        "monitor gpu": False,
        "Priority": 1,
        "Assays": [
            {
                "assay id": "Prey-Static-5",
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
                    "prey 1": {"steps": 1100,
                               "size": 5,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Static-10",
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
                    "prey 1": {"steps": 1100,
                               "size": 10,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Static-15",
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
                    "prey 1": {"steps": 1100,
                               "size": 15,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Left-5",
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
                    "prey 1": {"steps": 1100,
                               "size": 5,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Left-10",
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
                    "prey 1": {"steps": 1100,
                               "size": 10,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Left-15",
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
                    "prey 1": {"steps": 1100,
                               "size": 15,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Right-5",
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
                    "prey 1": {"steps": 1100,
                               "size": 5,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Right-10",
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
                    "prey 1": {"steps": 1100,
                               "size": 10,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Right-15",
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
                    "prey 1": {"steps": 1100,
                               "size": 15,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Away",
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
                    "prey 1": {"steps": 1100,
                               "size": 15,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
            {
                "assay id": "Prey-Towards",
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
                    "prey 1": {"steps": 1100,
                               "size": 5,
                               "interval": 100,
                               },
                },
                "interactions": []
            },
        ]
    },
]
