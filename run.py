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
    {
        "Model Name": "ppo_continuous_beta_sanity",
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
        "Model Name": "ppo_continuous_beta_sanity",
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
        "Model Name": "ppo_continuous_beta_sanity",
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

dqn_discrete = [
    {
        "Model Name": "dqn_discrete",
        "Environment Name": "dqn_discrete",
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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
        "Trial Number": 2,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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
        "Trial Number": 3,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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
        "Trial Number": 4,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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
        "Trial Number": 1,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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
        "Trial Number": 2,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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
        "Trial Number": 3,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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
        "Trial Number": 4,
        "Total Configurations": 6,
        "Episode Transitions": {
        },
        "Conditional Transitions": {
            "Prey Caught": {
                "2": 4,
                "3": 5,
                "4": 6,
                "5": 7,
                "6": 8,
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

print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
manager = TrialManager(ppo_continuous_mv_es, parallel_jobs=4)
manager.run_priority_loop()
