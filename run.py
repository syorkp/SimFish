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
    #         "Prey Caught": {
    #             "2": 5,
    #             "3": 6,
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
    #         "Prey Caught": {
    #             "2": 5,
    #             "3": 6,
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
    #         "Prey Caught": {
    #             "2": 5,
    #             "3": 6,
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
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
            "Prey Caught": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
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
            "Prey Caught": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
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
            "Prey Caught": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
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
            "Prey Caught": {
                "2": 3,
                "3": 4,
                "4": 5,
                "5": 6,
                "6": 7,
                "7": 8,
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
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_is",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_is",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_sbe_is",
        "Environment Name": "ppo_continuous_sbe_is",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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

ppo_continuous_sbe_es = [
    {
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_sbe_es",
        "Environment Name": "ppo_continuous_sbe_es",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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

ppo_continuous_mv_is = [
    {
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_mv_is",
        "Environment Name": "ppo_continuous_mv_is",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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

ppo_continuous_uv_is = [
    {
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 1,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_uv_is",
        "Environment Name": "ppo_continuous_uv_is",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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

ppo_beta_normal = [
    {
        "Model Name": "ppo_continuous_beta_normal",
        "Environment Name": "ppo_continuous_beta",
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
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_mv_es",
        "Environment Name": "ppo_continuous_mv_es",
        "Trial Number": 2,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_mv_es",
        "Environment Name": "ppo_continuous_mv_es",
        "Trial Number": 3,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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
        "Model Name": "ppo_continuous_mv_es",
        "Environment Name": "ppo_continuous_mv_es",
        "Trial Number": 4,
        "Total Configurations": 5,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 6,
                "4": 7,
                "5": 8,
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

data_gathering_config_continuous = [{
    "Model Name": "scaffold_version_4",
    "Environment Name": "continuous_assay",
    "Assay Configuration Name": "Behavioural-Data-Free",
    "Trial Number": 4,
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

    "Assays": [
        {
            "assay id": "Naturalistic-3",
            "stimulus paradigm": "Naturalistic",
            "duration": 1000,
            "Tethered": False,
            "save frames": False,
            "save stimuli": False,
            "random positions": False,
            "reset": False,
            "background": None,
            "moving": False,
            "collisions": True,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": [layer for layer in base_network_layers.keys()] + ["left_eye", "right_eye", "internal_state"],  # Same organisation as ablations
            "ablations": []
        },
        {
            "assay id": "Naturalistic-4",
            "stimulus paradigm": "Naturalistic",
            "duration": 1000,
            "Tethered": False,
            "save frames": False,
            "save stimuli": False,
            "random positions": False,
            "reset": False,
            "background": None,
            "moving": False,
            "collisions": True,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": [layer for layer in base_network_layers.keys()] + ["left_eye", "right_eye",
                                                                                     "internal_state"],
            # Same organisation as ablations
            "ablations": []
        },
        {
            "assay id": "Naturalistic-5",
            "stimulus paradigm": "Naturalistic",
            "duration": 1000,
            "Tethered": False,
            "save frames": False,
            "save stimuli": False,
            "random positions": False,
            "reset": False,
            "background": None,
            "moving": False,
            "collisions": True,
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": [layer for layer in base_network_layers.keys()] + ["left_eye", "right_eye",
                                                                                     "internal_state"],
            # Same organisation as ablations
            "ablations": []
        },
    ]
}]

data_gathering_config_discrete = [{
    "Model Name": "dqn_scaffold_version_3",
    "Environment Name": "discrete_dqn_assay",
    "Assay Configuration Name": "Behavioural-Data-Free",
    "Trial Number": 2,
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
            "recordings": [],
            "behavioural recordings": ["environmental positions", "observation"],
            "network recordings": [], #[layer for layer in base_network_layers.keys()] + ["left_eye", "right_eye", "internal_state"],  # Same organisation as ablations
            "ablations": []
        },
    ]
}]

ppo_scaffold_training_config_3a = [
    {
        "Model Name": "scaffold_version_3",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_3",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "scaffold_version_3",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_3",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

dqn_scaffold_training_config_3a = [
    {
        "Model Name": "dqn_scaffold_version_3",
        "Environment Name": "dqn_scaffold_3",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "dqn_scaffold_version_3",
        "Environment Name": "dqn_scaffold_3",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

dqn_scaffold_training_config_3b = [
    {
        "Model Name": "dqn_scaffold_version_3",
        "Environment Name": "dqn_scaffold_3",
        "Trial Number": 3,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "dqn_scaffold_version_3",
        "Environment Name": "dqn_scaffold_3",
        "Trial Number": 4,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

ppo_scaffold_training_config_5a = [
    {
        "Model Name": "scaffold_version_5",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_5",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "scaffold_version_5",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_5",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

ppo_scaffold_training_config_5a_no_entropy = [
    {
        "Model Name": "scaffold_version_5_ne",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_5_ne",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "scaffold_version_5_ne",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_5_ne",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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


ppo_scaffold_training_config_5b = [
    {
        "Model Name": "scaffold_version_5",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_5",
        "Trial Number": 3,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "scaffold_version_5",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_5",
        "Trial Number": 4,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

dqn_scaffold_training_config_5a = [
    {
        "Model Name": "dqn_scaffold_version_5",
        "Environment Name": "dqn_scaffold_5",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "dqn_scaffold_version_5",
        "Environment Name": "dqn_scaffold_5",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

dqn_scaffold_training_config_5b = [
    {
        "Model Name": "dqn_scaffold_version_5",
        "Environment Name": "dqn_scaffold_5",
        "Trial Number": 3,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "dqn_scaffold_version_5",
        "Environment Name": "dqn_scaffold_5",
        "Trial Number": 4,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

dqn_scaffold_training_config_5c = [
    {
        "Model Name": "dqn_scaffold_version_5",
        "Environment Name": "dqn_scaffold_5",
        "Trial Number": 5,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Profile Speed": True,
    },
    {
        "Model Name": "dqn_scaffold_version_5",
        "Environment Name": "dqn_scaffold_5",
        "Trial Number": 6,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Profile Speed": True,
    },
]

ppo_scaffold_training_config_7a = [
    {
        "Model Name": "scaffold_version_7",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_7",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "scaffold_version_7",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_7",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

ppo_scaffold_training_config_7b = [
    {
        "Model Name": "scaffold_version_7",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_7",
        "Trial Number": 3,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Using GPU": False,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
    {
        "Model Name": "scaffold_version_7",
        "Environment Name": "ppo_continuous_sbe_is_scaffold_7",
        "Trial Number": 4,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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


dqn_scaffold_training_config_7a = [
    {
        "Model Name": "dqn_scaffold_version_7",
        "Environment Name": "dqn_scaffold_7",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "dqn_scaffold_version_7",
        "Environment Name": "dqn_scaffold_7",
        "Trial Number": 2,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

dqn_scaffold_training_config_7b = [
    {
        "Model Name": "dqn_scaffold_version_7",
        "Environment Name": "dqn_scaffold_7",
        "Trial Number": 3,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Model Name": "dqn_scaffold_version_7",
        "Environment Name": "dqn_scaffold_7",
        "Trial Number": 4,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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

mod_test_config = [
    {
        "Model Name": "modular_network_test",
        "Environment Name": "modular_network_test",
        "Trial Number": 1,
        "Total Configurations": 8,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 5,
                "3": 5,
                "4": 6,
                "5": 6,
                "6": 7,
                "8": 8,
                "9": 8,
            },
            "Predators Avoided": {
                "7": 4
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
        "Using GPU": False,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True,
        "Profile Speed": False,
    },
]


print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(ppo_scaffold_training_config_7b, parallel_jobs=4)
manager.run_priority_loop()
