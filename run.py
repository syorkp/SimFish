import os
import sys

import json
from datetime import datetime
import numpy as np

from Services.trial_manager import TrialManager
from Configurations.Templates.interruptions import energy_state_profile_long_term, salt_profile_long_term, \
    in_light_profile_long_term, efference_A, efference_B, efference_C, efference_D, efference_E, efference_F, \
    efference_G, efference_H, efference_V, efference_W, efference_X, efference_Y, efference_Z, long_term_visual_interruption_profile, \
    brief_interruption_profile, fish_relocation_to_nowhere, energy_state_profile_long_term, long_term_interruption_profile, rnn_zeros

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

    with open("./Run-Configurations/VRV_CONFIG.json", "r") as file:
        vrv_config = json.load(file)

    vrv_config = vrv_config[0]
    vrv_config["Model Name"] = "dqn_scaffold_14"
    vrv_config["Trial Number"] = 1
    vrv_config["Checkpoint"] = None
    vrv_config["New Simulation"] = True
    vrv_config["Full Reafference"] = False
    for i, assay in enumerate(vrv_config["Assays"]):
        vrv_config["Assays"][i]["behavioural recordings"] = ["environmental positions", "observation"]
        vrv_config["Assays"][i]["network recordings"] = ["rnn state", "internal state"]
        vrv_config["Assays"][i]["use_mu"] = True
        vrv_config["Assays"][i]["energy_state_control"] = "Held"
        vrv_config["Assays"][i]["salt_control"] = False
    vrv_config["Using GPU"] = False
    vrv_config["Continuous Actions"] = False
    vrv_config["SB Emulator"] = True
    vrv_config["Learning Algorithm"] = "DQN"
    vrv_config["Assays"] = [vrv_config["Assays"][0]]
    vrv_config["Assays"][0]["save frames"] = True
    vrv_config["Assays"][0]["stimuli"]["prey 1"]["steps"] = 200
    vrv_config["Assays"][0]["duration"] = 200

    vrv_config = [vrv_config]

    # Ablation configs
    with open('Configurations/Ablation-Matrices/post_ablation_weights_1_dqn_26_2.npy', 'rb') as f:
        ablation_matrix = np.load(f)
    with open('Configurations/Ablation-Matrices/post_ablation_weights_2_dqn_26_2.npy', 'rb') as f:
        full_ablation_matrix = np.load(f)


   
    # NEW VISUAL SYSTEM

    dqn_easy_v1 = [
        {
            "Model Name": "dqn_easy_v1",
            "Environment Name": "dqn_easy_v1",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]
 
    dqn_easy_v2 = [
        {
            "Model Name": "dqn_easy_v2",
            "Environment Name": "dqn_easy_v2",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_easy_v3 = [
        {
            "Model Name": "dqn_easy_v3",
            "Environment Name": "dqn_easy_v3",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    dqn_easy_v4 = [
        {
            "Model Name": "dqn_easy_v4",
            "Environment Name": "dqn_easy_v4",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    salt_only_1 = [
        {
            "Model Name": "salt_only",
            "Environment Name": "salt_only",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    salt_only_2 = [
        {
            "Model Name": "salt_only",
            "Environment Name": "salt_only",
            "Trial Number": 2,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    salt_only_3 = [
        {
            "Model Name": "salt_only",
            "Environment Name": "salt_only",
            "Trial Number": 3,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    salt_only_4 = [
        {
            "Model Name": "salt_only",
            "Environment Name": "salt_only",
            "Trial Number": 4,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    salt_only_5 = [
        {
            "Model Name": "salt_only",
            "Environment Name": "salt_only",
            "Trial Number": 5,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    predator_only = [
        {
            "Model Name": "predator_only",
            "Environment Name": "predator_only",
            "Trial Number": 1,
            "Run Mode": "Training",
            "Learning Algorithm": "DQN",
        },
    ]

    if run_config is None:
        run_config = dqn_epsilon_6
    else:
        print(f"{run_config} entered.")
        run_config = globals()[run_config]

    print(f"Start time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    manager = TrialManager(run_config, parallel_jobs=1)
    manager.run_priority_loop()
