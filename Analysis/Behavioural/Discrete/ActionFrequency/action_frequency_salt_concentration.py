import numpy as np
import matplotlib.pyplot as plt

from Analysis.Behavioural.Tools.get_repeated_data_parameter import get_parameter_across_trials
from Analysis.Behavioural.Tools.BehavLabels.extract_photogradient_sequences import get_in_light_vs_dark_steps
from Analysis.Behavioural.VisTools.get_action_name import get_action_name



def compare_action_usage_over_salt_concentration(model_name, assay_config, assay_id, n):
    actions = get_parameter_across_trials(model_name, assay_config, assay_id, n, "action")
    salt_concentration = get_parameter_across_trials(model_name, assay_config, assay_id, n, "salt")

    salt_concentration_low = [sc < np.mean(np.concatenate(salt_concentration)) / 0.8 for sc in salt_concentration]
    salt_concentration_high = [sc >= np.mean(np.concatenate(salt_concentration)) / 0.8  for sc in salt_concentration]

    salt_concentration_low = np.concatenate(salt_concentration_low)
    salt_concentration_high = np.concatenate(salt_concentration_high)

    actions_in_light_present, actions_in_light_counts = np.unique(salt_concentration_low, return_counts=True)
    actions_in_dark_present, actions_in_dark_counts = np.unique(salt_concentration_high, return_counts=True)
    actions_present = list(set(np.concatenate((actions_in_dark_present, actions_in_light_present))))

    actions_light_dark_table_counts = np.zeros((len(actions_present), 2))
    actions_light_dark_table_counts[:, 0] = np.array(actions_present)

    light_in_grid = np.array(actions_present) == actions_in_light_present
    dark_in_grid = np.array(actions_present) == actions_in_dark_present

    actions_light_dark_table_counts[light_in_grid, 0] = actions_in_light_counts
    actions_light_dark_table_counts[dark_in_grid, 1] = actions_in_dark_counts

    actions_light_dark_table_counts /= np.sum(actions_light_dark_table_counts, axis=0)

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]

    fig, ax = plt.subplots(figsize=(20, 10))
    for i, a in enumerate(reversed(list(actions_present))):
        # total_light = sum([actions_light_dark_table_counts[int(i), 0] for i in range(int(a + 1))])
    #        total_dark = sum([actions_light_dark_table_counts[int(i), 1] for i in range(int(a + 1))])

        total_light = np.sum(actions_light_dark_table_counts[i:, 0])
        total_dark = np.sum(actions_light_dark_table_counts[i:, 1])
        ax.bar([i for i in range(2)], [total_light, total_dark], color=color_set[int(a)], width=0.4)

    plt.xticks([i for i in range(2)], ["Light", "Dark"], fontsize=20)
    plt.ylabel("Proportion of Actions", fontsize=30)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.legend([get_action_name(int(a)) for a in reversed(actions_present)])
    plt.show()



if __name__ == "__main__":
    compare_action_usage_over_salt_concentration("dqn_salt_only_reduced-1", "Behavioural-Data-Free", "Naturalistic", 40)
