import sys
import os
import json

import numpy as np

from Analysis.Neural.MEI.estimate_mei_direct import produce_meis, produce_meis_extended
from Analysis.Video.behaviour_video_construction import draw_episode
from Analysis.Video.neural_video_construction import create_network_video, convert_ops_to_graph
from Analysis.load_data import load_data
from Analysis.load_model_config import load_assay_configuration_files

try:
    run_config = sys.argv[1]
except IndexError:
    run_config = "draw_ep"


if run_config == "extended_1l":
    produce_meis_extended("dqn_scaffold_26-2", "conv1l", True, 1000)
elif run_config == "extended_2l":
    produce_meis_extended("dqn_scaffold_26-2", "conv2l", True, 1000)
elif run_config == "extended_3l":
    produce_meis_extended("dqn_scaffold_26-2", "conv3l", True, 1000)
elif run_config == "extended_4l":
    produce_meis_extended("dqn_scaffold_26-2", "conv4l", True, 1000)
elif run_config == "1l":
    produce_meis("dqn_scaffold_26-2", "conv1l", True, 1000)
elif run_config == "2l":
    produce_meis("dqn_scaffold_26-2", "conv2l", True, 1000)
elif run_config == "3l":
    produce_meis("dqn_scaffold_26-2", "conv3l", True, 1000)
elif run_config == "4l":
    produce_meis("dqn_scaffold_26-2", "conv4l", True, 1000)
elif run_config == "dense":
    produce_meis("dqn_scaffold_26-2", "rnn_in", full_efference_copy=True, iterations=100, conv=False)
elif run_config == "draw_ep":
    models = ["dqn_gamma_pm-2", "dqn_gamma_pm-3", "dqn_gamma_pm-4", "dqn_gamma_pm-5"]
    for model in models:
        for i in range(1, 6):
            data = load_data(model, "Behavioural-Data-Free", f"Naturalistic-{i}")
            assay_config_name = "dqn_gamma_final"
            save_location = f"Analysis-Output/Behavioural/Videos/{model}-{i}-behaviour"

            try:
                with open(f"../../Configurations/Assay-Configs/{assay_config_name}_env.json", 'r') as f:
                    env_variables = json.load(f)
            except FileNotFoundError:
                with open(f"Configurations/Assay-Configs/{assay_config_name}_env.json", 'r') as f:
                    env_variables = json.load(f)

            draw_episode(data, env_variables, save_location, continuous_actions=False, show_energy_state=False,
                         trim_to_fish=True, showed_region_quad=750, save_id=f"{i}", include_sediment=True,
                         as_gif=False, s_per_frame=0.1, scale=0.5)

else:
    produce_meis_extended("dqn_scaffold_26-2", "conv1l", True, 1000)





