import json
import os
from datetime import datetime

from Services.trial_manager import TrialManager

# Ensure output directories exist
if not os.path.exists("./Training-Output/"):
    os.makedirs("./Output/")

if not os.path.exists("./Assay-Output/"):
    os.makedirs("./Assay-Output/")

# Setting .nv location to prevent GPU error
if not os.path.exists("./GPU-Caches/"):
    os.makedirs("./GPU-Caches/")

directory = "./GPU-Caches/"
existing_caches = [os.path.join(o) for o in os.listdir(directory) if os.path.isdir(os.path.join(directory,o))]

if len(existing_caches) > 0:
    caches_as_int = [int(o) for o in existing_caches]
    last_cache_number = max(caches_as_int)
    os.makedirs(f"./GPU-Caches/{last_cache_number + 1}")
    os.environ["__GL_SHADER_DISK_CACHE_PATH"] = f"./GPU-Caches/{last_cache_number+1}/"
else:
    os.makedirs(f"./GPU-Caches/{1}")
    os.environ["__GL_SHADER_DISK_CACHE_PATH"] = f"./GPU-Caches/{1}/"


# Loading VRV configs

with open("./Run-Configurations/VRV_CONFIG.json", "r") as file:
    vrv_config = json.load(file)

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

phase_1_test_config_discrete = [
    {
        "Model Name": "ppo_jade_phase_1_test_discrete",
        "Environment Name": "ppo_discrete_sbe_new_simulation",
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
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True
    },
]

phase_1_test_config = [
    {
        "Model Name": "calibration_test_am",
        "Environment Name": "ppo_continuous_sbe_is_new_simulation",
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
        "SB Emulator": True,
        "New Simulation": True
    },
    {
        "Model Name": "calibration_test_am",
        "Environment Name": "ppo_continuous_sbe_is_new_simulation",
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
        "SB Emulator": True,
        "New Simulation": True
    },
]

calibration_test_config = [
    {
        "Model Name": "calibration_test_am_final",
        "Environment Name": "ppo_continuous_sbe_is_final_calib",
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
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True
    },
    {
        "Model Name": "calibration_test_am_final",
        "Environment Name": "ppo_continuous_sbe_is_final_calib",
        "Trial Number": 2,
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
        "Learning Algorithm": "PPO",
        "Priority": 1,
        "Using GPU": True,
        "monitor gpu": False,
        "Full Logs": True,
        "SB Emulator": True,
        "New Simulation": True
    },
]


print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(calibration_test_config, parallel_jobs=2)
manager.run_priority_loop()
