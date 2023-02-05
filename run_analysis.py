import sys
import os

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
    produce_meis("dqn_scaffold_26-2", "rnn_in", full_reafference=True, iterations=100, conv=False)
elif run_config == "draw_ep":
    models = ["dqn_gamma_pm-2", "dqn_gamma_pm-3", "dqn_gamma_pm-4", "dqn_gamma_pm-5"]
    models = ["dqn_gamma-3"]
    for model in models:
        for i in range(1, 51):
            data = load_data(model, "Behavioural-Data-Free", f"Naturalistic-{i}")
            assay_config_name = "dqn_gamma_final"
            draw_episode(data, assay_config_name, model, continuous_actions=False, show_energy_state=False,
                         trim_to_fish=True, showed_region_quad=750, save_id=f"{i}", include_background=True,
                         as_gif=False, s_per_frame=0.1, scale=0.5)

    # model_name = "dqn_scaffold_33-1"
    # data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-1")
    # assay_config_name = "dqn_33_1"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="background", include_background=True)
    #
    # model_name = "dqn_scaffold_33-1"
    # data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-2")
    # assay_config_name = "dqn_33_1"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="background", include_background=True)
    #
    # model_name = "dqn_scaffold_33-1"
    # data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-3")
    # assay_config_name = "dqn_33_1"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="background", include_background=True)
    #
    # model_name = "dqn_scaffold_33-1"
    # data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-4")
    # assay_config_name = "dqn_33_1"
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="background", include_background=True)

    # model_name = "xdqn_scaffold_14-2"
    # assay_config_name = "dqn_26_2_videos"
    #
    # data = load_data(model_name, "Behavioural-Data-Videos-A1", "Naturalistic-1")
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="A11")
    # data = load_data(model_name, "Behavioural-Data-Videos-C1", "Naturalistic-1")
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="C11")

    # model_name = "dqn_scaffold_26-2"
    # assay_config_name = "dqn_26_2_videos"
    #
    # data = load_data(model_name, "Behavioural-Data-Videos-CONV", "Naturalistic-2")
    #
    # learning_params, environment_params, base_network_layers, ops, connectivity = load_configuration_files(model_name)
    # base_network_layers["rnn_state_actor"] = base_network_layers["rnn"]
    # del base_network_layers["rnn"]
    # network_data = {key: data[key] for key in list(base_network_layers.keys())}
    # network_data["rnn"] = data["rnn_state_actor"][:, 0, 0, :]
    # base_network_layers["rnn"] = base_network_layers["rnn_state_actor"]
    # del base_network_layers["rnn_state_actor"]
    # del network_data["rnn_state_actor"]
    #
    # network_data["left_eye"] = data["observation"][:, :, :, 0]
    # network_data["right_eye"] = data["observation"][:, :, :, 1]
    # network_data["internal_state"] = np.concatenate((np.expand_dims(data["energy_state"], 1),
    #                                                  np.expand_dims(data["salt"], 1)), axis=1)
    #
    # ops = convert_ops_to_graph(ops)
    # create_network_video(network_data, connectivity + ops, model_name, save_id="CONV", s_per_frame=0.04, scale=1)
    #
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              draw_past_actions=True,
    #              trim_to_fish=True, showed_region_quad=750, save_id="CONV", s_per_frame=0.04)

    # model_name = "ppo_scaffold_21-2"
    # assay_config_name = "ppo_21_2_videos"
    #
    # data = load_data(model_name, "Behavioural-Data-Videos-A1", "Naturalistic-5")
    # draw_episode(data, assay_config_name, model_name, continuous_actions=True, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="A15")
    # data = load_data(model_name, "Behavioural-Data-Videos-B1", "Naturalistic-3")
    # draw_episode(data, assay_config_name, model_name, continuous_actions=True, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="C11")
else:
    produce_meis_extended("dqn_scaffold_26-2", "conv1l", True, 1000)





