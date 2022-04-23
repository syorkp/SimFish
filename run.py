import os
import shutil

# Remove GPU Cache
if os.path.exists("../.nv"):
    print("Directory exists, removed it")
    shutil.rmtree("../.nv")
else:
    print("Directory didnt exist")
    d = '..'
    print([os.path.join(d, o) for o in os.listdir(d)
           if os.path.isdir(os.path.join(d, o))])
import json
from datetime import datetime

from Services.trial_manager import TrialManager
from Configurations.Networks.original_network import base_network_layers, ops, connectivity

# Ensure output directories exist
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Training-Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

# Setting .nv location to prevent GPU error
# if not os.path.exists("./GPU-Caches/"):
#     os.makedirs("./GPU-Caches/")
#
# directory = "./GPU-Caches/"
# existing_caches = [os.path.join(o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))]
#
# if len(existing_caches) > 0:
#     caches_as_int = [int(o) for o in existing_caches]
#     last_cache_number = max(caches_as_int)
#     os.makedirs(f"./GPU-Caches/{last_cache_number + 1}")
#     os.environ["__GL_SHADER_DISK_CACHE_PATH"] = f"./GPU-Caches/{last_cache_number+1}/"
# else:
#     os.makedirs(f"./GPU-Caches/{1}")
#     os.environ["__GL_SHADER_DISK_CACHE_PATH"] = f"./GPU-Caches/{1}/"


# Loading VRV configs

# with open("./Run-Configurations/VRV_CONFIG.json", "r") as file:
#     vrv_config = json.load(file)

ppo_beta_configuration = [
    {
        "Model Name": "ppo_continuous_beta_sanity",
        "Environment Name": "ppo_continuous_beta",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    # {
    #     "Model Name": "ppo_continuous_beta_sanity",
    #     "Environment Name": "ppo_continuous_beta",
    #     "Trial Number": 2,
    #     "Total Configurations": 3,
    #     "Episode Transitions": {
    #     },
    #     "Conditional Transitions": {
    #         "Prey Capture Index": {
    #             "2": 5,
    #             "3": 6,
    #         },
    #         "Predator Avoidance Index": {
    #         },
    #         "Sand Grains Bumped": {
    #         }
    #     },
    #     "Run Mode": "Training",
    #     "Tethered": False,
    #     "Realistic Bouts": True,
    #     "Continuous Actions": True,
    #     "Learning Algorithm": "PPO",
    #     "Priority": 1,
    #     "Using GPU": True,
    #     "monitor gpu": False,
    #     "Full Logs": True,
    #     "SB Emulator": False
    # },
    # {
    #     "Model Name": "ppo_continuous_beta_sanity",
    #     "Environment Name": "ppo_continuous_beta",
    #     "Trial Number": 3,
    #     "Total Configurations": 3,
    #     "Episode Transitions": {
    #     },
    #     "Conditional Transitions": {
    #         "Prey Capture Index": {
    #             "2": 5,
    #             "3": 6,
    #         },
    #         "Predator Avoidance Index": {
    #         },
    #         "Sand Grains Bumped": {
    #         }
    #     },
    #     "Run Mode": "Training",
    #     "Tethered": False,
    #     "Realistic Bouts": True,
    #     "Continuous Actions": True,
    #     "Learning Algorithm": "PPO",
    #     "Priority": 1,
    #     "Using GPU": True,
    #     "monitor gpu": False,
    #     "Full Logs": True,
    #     "SB Emulator": False
    # },
    # {
    #     "Model Name": "ppo_continuous_beta_sanity",
    #     "Environment Name": "ppo_continuous_beta",
    #     "Trial Number": 4,
    #     "Total Configurations": 3,
    #     "Episode Transitions": {
    #     },
    #     "Conditional Transitions": {
    #         "Prey Capture Index": {
    #             "2": 5,
    #             "3": 6,
    #         },
    #         "Predator Avoidance Index": {
    #         },
    #         "Sand Grains Bumped": {
    #         }
    #     },
    #     "Run Mode": "Training",
    #     "Tethered": False,
    #     "Realistic Bouts": True,
    #     "Continuous Actions": True,
    #     "Learning Algorithm": "PPO",
    #     "Priority": 1,
    #     "Using GPU": True,
    #     "monitor gpu": False,
    #     "Full Logs": True,
    #     "SB Emulator": False
    # },
]

dqn_discrete = [
    {
        "Model Name": "dqn_discrete",
        "Environment Name": "dqn_discrete",
        "Trial Number": 5,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "dqn_discrete",
        "Environment Name": "dqn_discrete",
        "Trial Number": 6,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "dqn_discrete",
        "Environment Name": "dqn_discrete",
        "Trial Number": 7,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "dqn_discrete",
        "Environment Name": "dqn_discrete",
        "Trial Number": 8,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },

]

ppo_discrete_sbe = [
    {
        "Model Name": "ppo_discrete_sbe",
        "Environment Name": "ppo_discrete_sbe",
        "Trial Number": 5,
        "Total Configurations": 7,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
            },
            "Predator Avoidance Index": {
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
        "Full Logs": True,
        "SB Emulator": True
    },
    {
        "Model Name": "ppo_discrete_sbe",
        "Environment Name": "ppo_discrete_sbe",
        "Trial Number": 6,
        "Total Configurations": 7,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
            },
            "Predator Avoidance Index": {
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
        "Full Logs": True,
        "SB Emulator": True
    },
    {
        "Model Name": "ppo_discrete_sbe",
        "Environment Name": "ppo_discrete_sbe",
        "Trial Number": 7,
        "Total Configurations": 7,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
            },
            "Predator Avoidance Index": {
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
        "Full Logs": True,
        "SB Emulator": True
    },
    {
        "Model Name": "ppo_discrete_sbe",
        "Environment Name": "ppo_discrete_sbe",
        "Trial Number": 8,
        "Total Configurations": 7,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
            },
            "Predator Avoidance Index": {
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
        "Full Logs": True,
        "SB Emulator": True
    },
]

ppo_continuous_sbe_is = [
    {
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_is",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_is",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_is",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_is",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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

ppo_continuous_sbe_es = [
    {
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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

ppo_continuous_mv_is = [
    {
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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

ppo_continuous_uv_is = [
    {
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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

ppo_beta_normal = [
    {
        "Model Name": "ppo_continuous_beta_normal",
        "Environment Name": "ppo_continuous_beta",
        "Trial Number": 1,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "ppo_continuous_beta_normal",
        "Environment Name": "ppo_continuous_beta",
        "Trial Number": 2,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "ppo_continuous_beta_normal",
        "Environment Name": "ppo_continuous_beta",
        "Trial Number": 3,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
    {
        "Model Name": "ppo_continuous_beta_normal",
        "Environment Name": "ppo_continuous_beta",
        "Trial Number": 4,
        "Total Configurations": 3,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False
    },
]

ppo_continuous_mv_es = [
    {
        "Model Name": "ppo_continuous_mv_es",
        "Environment Name": "ppo_continuous_mv_es",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_mv_es",
        "Environment Name": "ppo_continuous_mv_es",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_mv_es",
        "Environment Name": "ppo_continuous_mv_es",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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
        "Model Name": "ppo_continuous_mv_es",
        "Environment Name": "ppo_continuous_mv_es",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
            },
            "Predator Avoidance Index": {
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

ppo_data_gathering = [
    {
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_predators",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "set random seed": False,
        "Assays": [
            {
                "assay id": "Naturalistic-6",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-7",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-8",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-9",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-10",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-11",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-12",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-13",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-14",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
            {
                "assay id": "Naturalistic-15",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "recordings": ["rnn state", "environmental positions"],
                "ablations": []
            },
        ]
    },
]


# Ignore above

ppo_scaffold_training_config_8a_on = [
    {
        "Model Name": "ppo_scaffold_version_on_8",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_version_on_8",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_8b_on = [
    {
        "Model Name": "ppo_scaffold_version_on_8",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on",
        "Trial Number": 3,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_version_on_8",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on",
        "Trial Number": 4,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_8a_on_se = [
    {
        "Model Name": "ppo_scaffold_version_on_8_se",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on_se",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_version_on_8_se",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on_se",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_8b_on_se = [
    {
        "Model Name": "ppo_scaffold_version_on_8_se",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on_se",
        "Trial Number": 3,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_version_on_8_se",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_8_on_se",
        "Trial Number": 4,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "8": 0.4,
                "9": 0.5,
            },
            "Predator Avoidance Index": {
                "7": 400
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

dqn_scaffold_training_config_9a = [
    {
        "Model Name": "dqn_scaffold_9",
        "Environment Name": "dqn_scaffold_9",
        "Trial Number": 1,
        "Total Configurations": 11,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_9",
        "Environment Name": "dqn_scaffold_9",
        "Trial Number": 2,
        "Total Configurations": 11,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

dqn_scaffold_training_config_9b = [
    {
        "Model Name": "dqn_scaffold_9",
        "Environment Name": "dqn_scaffold_9",
        "Trial Number": 3,
        "Total Configurations": 11,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_9",
        "Environment Name": "dqn_scaffold_9",
        "Trial Number": 4,
        "Total Configurations": 11,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_9a = [
    {
        "Model Name": "ppo_scaffold_low_sig_9",
        "Environment Name": "ppo_scaffold_low_sig_9",
        "Trial Number": 1,
        "Total Configurations": 13,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
                "12": 0.5,
                "13": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_low_sig_9",
        "Environment Name": "ppo_scaffold_low_sig_9",
        "Trial Number": 2,
        "Total Configurations": 13,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
                "12": 0.5,
                "13": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_9b = [
    {
        "Model Name": "ppo_scaffold_low_sig_9",
        "Environment Name": "ppo_scaffold_low_sig_9",
        "Trial Number": 3,
        "Total Configurations": 13,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
                "12": 0.5,
                "13": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_low_sig_9",
        "Environment Name": "ppo_scaffold_low_sig_9",
        "Trial Number": 4,
        "Total Configurations": 13,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Capture Index": {
                "2": 0.2,
                "3": 0.2,
                "4": 0.3,
                "5": 0.3,
                "6": 0.4,
                "7": 0.4,
                "8": 0.4,
                "9": 0.5,
                "10": 0.5,
                "11": 0.5,
                "12": 0.5,
                "13": 0.5,
            },
            "Predator Avoidance Index": {
            },
            "Sand Grains Bumped": {
            }
        },
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

# Scaffold 10 (and new architecture)

dqn_scaffold_training_config_10a = [
    {
        "Model Name": "dqn_scaffold_10",
        "Environment Name": "dqn_scaffold_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_10",
        "Environment Name": "dqn_scaffold_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

dqn_scaffold_training_config_10b = [
    {
        "Model Name": "dqn_scaffold_10",
        "Environment Name": "dqn_scaffold_10",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_10",
        "Environment Name": "dqn_scaffold_10",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_10a = [
    {
        "Model Name": "ppo_scaffold_eg_10",
        "Environment Name": "ppo_scaffold_eg_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_eg_10",
        "Environment Name": "ppo_scaffold_eg_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_10b = [
    {
        "Model Name": "ppo_scaffold_eg_10",
        "Environment Name": "ppo_scaffold_eg_10",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_eg_10",
        "Environment Name": "ppo_scaffold_eg_10",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_high_sigma_10a = [
    {
        "Model Name": "ppo_scaffold_hs_10",
        "Environment Name": "ppo_scaffold_hs_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_hs_10",
        "Environment Name": "ppo_scaffold_hs_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_high_sigma_10b = [
    {
        "Model Name": "ppo_scaffold_hs_10",
        "Environment Name": "ppo_scaffold_hs_10",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_hs_10",
        "Environment Name": "ppo_scaffold_hs_10",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_egf_10a = [
    {
        "Model Name": "ppo_scaffold_egf_10",
        "Environment Name": "ppo_scaffold_egf_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_egf_10",
        "Environment Name": "ppo_scaffold_egf_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_egf_10b = [
    {
        "Model Name": "ppo_scaffold_egf_10",
        "Environment Name": "ppo_scaffold_egf_10",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_egf_10",
        "Environment Name": "ppo_scaffold_egf_10",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

dqn_scaffold_training_config_dn_10a = [
    {
        "Model Name": "dqn_scaffold_dn_10",
        "Environment Name": "dqn_scaffold_dn_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_dn_10",
        "Environment Name": "dqn_scaffold_dn_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

dqn_scaffold_training_config_dn_10b = [
    {
        "Model Name": "dqn_scaffold_dn_10",
        "Environment Name": "dqn_scaffold_dn_10",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_dn_10",
        "Environment Name": "dqn_scaffold_dn_10",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

dqn_scaffold_training_config_11a = [
    {
        "Model Name": "dqn_scaffold_11",
        "Environment Name": "dqn_scaffold_11",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_11",
        "Environment Name": "dqn_scaffold_11",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

dqn_scaffold_training_config_11b = [
    {
        "Model Name": "dqn_scaffold_11",
        "Environment Name": "dqn_scaffold_11",
        "Trial Number": 3,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "dqn_scaffold_11",
        "Environment Name": "dqn_scaffold_11",
        "Trial Number": 4,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": False,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_egf_min_10a = [
    {
        "Model Name": "ppo_scaffold_egf_min_10",
        "Environment Name": "ppo_scaffold_egf_min_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_egf_min_10",
        "Environment Name": "ppo_scaffold_egf_min_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_egf_max_10a = [
    {
        "Model Name": "ppo_scaffold_egf_max_10",
        "Environment Name": "ppo_scaffold_egf_max_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_egf_max_10",
        "Environment Name": "ppo_scaffold_egf_max_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

ppo_scaffold_training_config_egf_energy_10a = [
    {
        "Model Name": "ppo_scaffold_egf_energy_10",
        "Environment Name": "ppo_scaffold_egf_energy_10",
        "Trial Number": 1,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "ppo_scaffold_egf_energy_10",
        "Environment Name": "ppo_scaffold_egf_energy_10",
        "Trial Number": 2,
        "Run Mode": "Training",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]

# Assay Configs

dqn_testing = [
    {
        "Model Name": "dqn_scaffold_10",
        "Environment Name": "dqn_scaffold_10_assay",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 3,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": False,
        "Learning Algorithm": "DQN",
        "Priority": 2,
        "Using GPU": False,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "set random seed": False,
        "New Simulation": True,
        "Profile Speed": False,
        "Assays": [
            {
                "assay id": "Naturalistic-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 500,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "use_mu": False,
                "recordings": ["action", "position", "prey_consumed", "prey_positions", "rnn state"],
                "behavioural recordings": ["environmental positions"],
                "network recordings": ["rnn state"],
                "ablations": []
            },
        ]
    },
]

ppo_testing = [
    {
        "Model Name": "ppo_scaffold_version_on_8_se",
        "Environment Name": "ppo_scaffold_version_on_8_se_assay",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": False,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "set random seed": False,
        "New Simulation": True,
        "Profile Speed": False,
        "Assays": [
            {
                "assay id": "Naturalistic-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 1000,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "use_mu": True,
                "behavioural recordings": ["environmental positions"],
                "network recordings": ["rnn state"],
                "ablations": []
            },
        ]
    },
]

ppo_testing_normal_sigma = [
    {
        "Model Name": "ppo_scaffold_version_on_8_se",
        "Environment Name": "ppo_scaffold_version_on_8_se_assay",
        "Assay Configuration Name": "Behavioural-Data-Free",
        "Trial Number": 1,
        "Run Mode": "Assay",
        "Tethered": False,
        "Realistic Bouts": True,
        "Continuous Actions": True,
        "Learning Algorithm": "PPO",
        "Priority": 2,
        "Using GPU": False,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "set random seed": False,
        "New Simulation": True,
        "Profile Speed": False,
        "Assays": [
            {
                "assay id": "NaturalisticNormalSigma-1",
                "stimulus paradigm": "Naturalistic",
                "duration": 500,
                "Tethered": False,
                "save frames": True,
                "save stimuli": False,
                "random positions": False,
                "reset": False,
                "background": None,
                "moving": False,
                "collisions": True,
                "use_mu": False,
                "behavioural recordings": ["environmental positions"],
                "network recordings": ["rnn state"],
                "ablations": []
            },
        ]
    },
]

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(dqn_scaffold_training_config_10b, parallel_jobs=4)
manager.run_priority_loop()
