import os
import sys

import json
from datetime import datetime
import numpy as np

from Services.trial_manager import TrialManager

if __name__ == "__main__": # may be needed to run on windows
    # Get config argument
    try:
        run_config = sys.argv[1]
    except IndexError:
        run_config = None

    # Ensure output directories exist
    if not os.path.exists("./Training-Output/"):
        os.makedirs("./Training-Output/")

    if not os.path.exists("./Assay-Output/"):
        os.makedirs("./Assay-Output/")

    #                   TRAINING - DQN

    local_test = [
        {
            "Model Name": "local_test",
            "Environment Name": "local_test",
            "Trial Number": 1,
            "Using GPU": False,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
    local_test_large = [
        {
            "Model Name": "local_test_large",
            "Environment Name": "local_test_large",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_salt_only_reduced = [
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
        {
            "Model Name": "dqn_salt_only_reduced",
            "Environment Name": "dqn_salt_only_reduced",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    #                   TRAINING - PPO

    ppo_proj = [
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
        {
            "Model Name": "ppo_proj",
            "Environment Name": "ppo_proj",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "PPO",
        },
    ]

    if run_config is None:
        run_config = dqn_salt_only_reduced
    else:
        print(f"{run_config} entered.")
        run_config = globals()[run_config]

    print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    manager = TrialManager(run_config, parallel_jobs=5)
    manager.run_priority_loop()
