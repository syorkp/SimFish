import matplotlib.pyplot as plt
import numpy as np

from Analysis.load_data import load_data


def compare_action_usage_over_time(data, bins=5):
    bin_groups = np.linspace(0, len(data["action"]), bins+1, endpoint=True).astype(int)
    action_counts = np.zeros((10, bins))
    actions_present = np.array([])

    for i, b in enumerate(bin_groups):
        if i == len(bin_groups)-1:
            pass
        else:
            actions = data["action"][b:bin_groups[i+1]]
            unique, counts = np.unique(actions, return_counts=True)
            actions_present = np.concatenate((actions_present, unique))
            for j, a in enumerate(unique):
                action_counts[a, i] += counts[j]

    action_proportions = action_counts / np.sum(action_counts[:, 0])
    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]
    actions_present = list(set(actions_present))

    fig, ax = plt.subplots(figsize=(20, 10))
    for a in reversed(list(actions_present)):
        total = sum([action_proportions[int(i)] for i in range(int(a + 1))])
        ax.bar([i for i in range(bins)], total,  color=color_set[int(a)], width=0.4)

    plt.xticks([i for i in range(bins)], [f"{bin_groups[i]}-{bin_groups[i+1]}" for i in range(bins)], fontsize=20)
    plt.ylabel("Proportion of Actions", fontsize=30)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    # plt.legend([get_action_name(int(a)) for a in reversed(actions_present)])
    plt.show()


def compare_action_usage_over_time_multiple_models(datas):
    ...


if __name__ == "__main__":
    data = load_data("dqn_scaffold_26-2", "Behavioural-Data-NaturalisticA", f"Naturalistic-3")
    compare_action_usage_over_time(data, bins=4)






