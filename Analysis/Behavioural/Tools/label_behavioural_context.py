import numpy as np


from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files

from Analysis.Behavioural.Tools.extract_capture_sequences import label_capture_sequences
from Analysis.Behavioural.Tools.extract_escape_sequences import label_escape_sequences
from Analysis.Behavioural.Tools.extract_exploration_sequences import label_exploration_sequences_no_prey, \
    label_exploration_sequences_free_swimming
from Analysis.Behavioural.Tools.extract_wall_interaction_sequences import label_wall_interaction_steps
from Analysis.Behavioural.Tools.extract_photogradient_sequences import label_in_light_steps, label_hemispheric_light_gradient
from Analysis.Behavioural.Tools.extract_salt_interaction_sequences import label_salt_health_decreasing

def get_behavioural_context_name_by_index(index):
    if index == 0:
        return "No Recognised Context"
    elif index == 1:
        return "Prey Capture"
    elif index == 2:
        return "Escape"
    elif index == 3:
        return "Exploration - No prey stimuli"
    elif index == 4:
        return "Exploration - Free swimming"
    elif index == 5:
        return "Wall interaction"
    elif index == 6:
        return "In light"
    elif index == 7:
        return "Directional brightness (left is event)"
    elif index == 8:
        return "Salt health decreasing"
    else:
        return "ERROR - UNKNOWN"


def label_behavioural_context(data, model_name):
    """Return list of lists of labels for behavioural context at each step.
        - 0 No recognised context
        - 1 Prey capture behaviour
        - 2 Escape behaviour
        - 3 Exploration behaviour
        - 4 Exploration behaviour 2
        - 5 Wall interaction
        - 6 Directional brightness (when left hemisphere is brighter, encoded with 1)
    """
    # TODO: load environment data and use below.

    learning_params, env_variables, _, _, _ = load_configuration_files(model_name)

    capture_ts = label_capture_sequences(data, n=20) * 1
    predator_avoidance_ts = label_escape_sequences(data) * 1
    exploration_np_ts = label_exploration_sequences_no_prey(data) * 1
    exploration_fs_ts = label_exploration_sequences_free_swimming(data, environment_size=env_variables["width"]) * 1
    wall_interaction_ts = label_wall_interaction_steps(data, 100, env_variables["width"]) * 1
    in_light_ts = label_in_light_steps(data) * 1
    directional_brightness_ts = label_hemispheric_light_gradient(data, env_variables) * 1
    salt_decreasing_ts = label_salt_health_decreasing(data, env_variables) * 1

    no_recognised_context = (capture_ts + exploration_fs_ts + predator_avoidance_ts + exploration_np_ts +
                             wall_interaction_ts + in_light_ts + directional_brightness_ts + salt_decreasing_ts) == 0

    no_recognised_context = np.expand_dims(no_recognised_context, 1)
    capture_ts = np.expand_dims(capture_ts, 1)
    predator_avoidance_ts = np.expand_dims(predator_avoidance_ts, 1)
    exploration_np_ts = np.expand_dims(exploration_np_ts, 1)
    exploration_fs_ts = np.expand_dims(exploration_fs_ts, 1)
    wall_interaction_ts = np.expand_dims(wall_interaction_ts, 1)
    in_light_ts = np.expand_dims(in_light_ts, 1)
    directional_brightness_ts = np.expand_dims(directional_brightness_ts, 1)
    salt_decreasing_ts = np.expand_dims(salt_decreasing_ts, 1)

    behavioural_context_label = np.concatenate((no_recognised_context, capture_ts, predator_avoidance_ts,
                                                exploration_np_ts, exploration_fs_ts, wall_interaction_ts, in_light_ts,
                                                directional_brightness_ts, salt_decreasing_ts), axis=1)
    return behavioural_context_label


def label_behavioural_context_multiple_trials(datas, model_name):
    associated_behavioural_context_labels = []

    for data in datas:
        associated_behavioural_context_labels.append(label_behavioural_context(data, model_name))

    return associated_behavioural_context_labels


if __name__ == "__main__":
    data = load_data("dqn_scaffold_18-2", "Behavioural-Data-Free", f"Naturalistic-2")

    context = label_behavioural_context(data, 1500, "dqn_scaffold_18-2")





