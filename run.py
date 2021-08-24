import os
from datetime import datetime
import json

from Services.trial_manager import TrialManager

# Ensure Output File Exists
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")


ppo_test_config = [
    {
        "Model Name": "ppo_test",
        "Environment Name": "ppo_test",
        "Trial Number": 2,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
    },
    {
        "Model Name": "ppo_test",
        "Environment Name": "ppo_test",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
    },
    {
        "Model Name": "ppo_test",
        "Environment Name": "ppo_test",
        "Trial Number": 4,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
    },
]

scaled_sigmas_configuration = [
    {
        "Model Name": "scaled_sigmas_configuration",
        "Environment Name": "continuous_learning_scaffold",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
    },
]

reward_penalties_configuration = [
    {
        "Model Name": "reward_penalties_test",
        "Environment Name": "continuous_with_reward_penalties",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
    },
    {
        "Model Name": "reward_penalties_test",
        "Environment Name": "continuous_with_reward_penalties",
        "Trial Number": 2,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
    },
    {
        "Model Name": "reward_penalties_test",
        "Environment Name": "continuous_with_reward_penalties",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 10,
                "3": 15,
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
    },
]

test_continuous_assay_configuration = [
    {
        "Model Name": "scaffold_test",
        "Environment Name": "continuous_assay",
        "Trial Number": 19,
        "Assay Configuration Name": "Checking_Observation",
        "Total Configurations": 3,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "set random seed": False,
        "Assays": [
            {
                "assay id": "Environment-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 2000,
                "Tethered": False,
                "save frames": True,
                "random positions": False,
                "background": None,
                "moving": False,
                "save stimuli": False,
                "reset": False,
                "collisions": True,

                "recordings": ["left_conv_4", "rnn state", "consumed", "impulse", "angle", "position"],
                "ablations": []
            },
        ]
    }
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(ppo_test_config)
manager.run_priority_loop()
