import os
from datetime import datetime
import json

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")


test_continuous_training_configuration = [
    # {
    #     "Model Name": "continuous_fast_learning",
    #     "Environment Name": "continuous_fast_learning",
    #     "Trial Number": 1,
    #     "Total Configurations": 1,
    #     "Episode Transitions": {
    #     },
    #     "Conditional Transitions": {
    #         "Prey Caught": {
    #         },
    #         "Predators Avoided": {
    #         },
    #         "Sand Grains Bumped": {
    #         }
    #     },
    #     "Run Mode": "Training",
    #     "Tethered": False,
    #     "Realistic Bouts": True,
    #     "Continuous Actions": True,
    #     "Priority": 2,
    #     "Using GPU": True,
    #     "monitor gpu": False,
    # },
    {
        "Model Name": "continuous_slow_learning",
        "Environment Name": "continuous_slow_learning",
        "Trial Number": 1,
        "Total Configurations": 1,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
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
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
    }
]

even_training_configuration = [
    {
        "Model Name": "new_even_prey_ref",
        "Environment Name": "even_prey",
        "Trial Number": 5,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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
        "Model Name": "new_even_prey_ref",
        "Environment Name": "even_prey",
        "Trial Number": 6,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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
        "Model Name": "new_even_prey_ref",
        "Environment Name": "even_prey",
        "Trial Number": 7,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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
        "Model Name": "new_even_prey_ref",
        "Environment Name": "even_prey",
        "Trial Number": 8,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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

differential_training_configuration = [
    {
        "Model Name": "new_differential_prey_ref",
        "Environment Name": "differential_prey",
        "Trial Number": 3,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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
        "Model Name": "new_differential_prey_ref",
        "Environment Name": "differential_prey",
        "Trial Number": 4,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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
        "Model Name": "new_differential_prey_ref",
        "Environment Name": "differential_prey",
        "Trial Number": 5,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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
        "Model Name": "new_differential_prey_ref",
        "Environment Name": "differential_prey",
        "Trial Number": 6,
        "Total Configurations": 4,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 8,
                "4": 10,
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

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(test_continuous_training_configuration)
manager.run_priority_loop()
