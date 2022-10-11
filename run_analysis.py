import sys
import os
from Analysis.Neural.MEI.estimate_mei_direct import produce_meis, produce_meis_extended
from Analysis.Video.behaviour_video_construction import draw_episode
from Analysis.Video.neural_video_construction import create_network_video, convert_ops_to_graph
from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files

try:
    run_config = sys.argv[1]
except IndexError:
    run_config = "1l"


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
    # model_name = "dqn_scaffold_26-2"
    # assay_config_name = "dqn_26_2_videos"
    #
    # data = load_data(model_name, "Behavioural-Data-Videos-A1", "Naturalistic-1")
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="A11")
    # data = load_data(model_name, "Behavioural-Data-Videos-C1", "Naturalistic-1")
    # draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
    #              trim_to_fish=True, showed_region_quad=750, save_id="C11")

    model_name = "dqn_scaffold_26-2"
    assay_config_name = "dqn_26_2_videos"

    data = load_data(model_name, "Behavioural-Data-Videos-CONV", "Naturalistic-2")
    draw_episode(data, assay_config_name, model_name, continuous_actions=False, show_energy_state=False,
                 trim_to_fish=True, showed_region_quad=750, save_id="CONV", s_per_frame=0.06)

    learning_params, environment_params, base_network_layers, ops, connectivity = load_configuration_files(model_name)
    network_data = {key: data[key] for key in list(base_network_layers.keys()) + ["left_eye", "right_eye"] + ["internal_state"]}
    ops = convert_ops_to_graph(ops)
    create_network_video(network_data, connectivity + ops, model_name, save_id="CONV", s_per_frame=0.06)

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





