import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Patch

from Analysis.load_data import load_data
from Analysis.Behavioural.VisTools.get_action_name import get_action_name


def get_provided_efference(letter):
    if letter == "A":
        action = "Slow2"
    elif letter == "B":
        action = "RT Left"
    elif letter == "C":
        action = "RT Right"
    elif letter == "D":
        action = "sCS"
    elif letter == "E":
        action = "J-turn Left"
    elif letter == "F":
        action = "J-turn Right"
    elif letter == "G":
        action = "Rest"
    elif letter == "H":
        action = "AS"
    elif letter == "V":
        action = "I=1, A=0"
    elif letter == "W":
        action = "I=4, A=0"
    elif letter == "X":
        action = "I=10, A=0"
    elif letter == "Y":
        action = "I=5, A=-0.3"
    elif letter == "Z":
        action = "I=5, A=0.3"
    else:
        action = "Error"
    return action


def compute_action_heterogeneity_continuous(impulses, angles, max_i, max_a):
    impulse_bins = np.linspace(0, max_i, 11)
    impulses_binned = np.zeros((10))

    angle_bins = np.linspace(-max_a, max_a, 11)
    angles_binned = np.zeros((10))

    for i in range(10):
        within_bin = (impulse_bins[i] <= impulses) * (impulses < impulse_bins[i+1]) * 1
        impulses_binned[i] += np.sum(within_bin)

        within_bin = (angle_bins[i] <= angles) * (angles < angle_bins[i+1]) * 1
        angles_binned[i] += np.sum(within_bin)

    impulse_heterogeneity = compute_heterogeneity(impulses_binned, impulses.shape[0])
    angle_heterogeneity = compute_heterogeneity(angles_binned, angles.shape[0])

    return impulse_heterogeneity, angle_heterogeneity


def compute_heterogeneity(frequencies, n_actions):
    differences = [np.max(np.delete(frequencies, i) - frequencies[i]) for i in range(frequencies.shape[0])]
    difference_sum = np.sum(np.absolute(differences))

    heterogeneity = n_actions/difference_sum
    heterogeneity -= 0.1
    heterogeneity /= n_actions/10
    heterogeneity = heterogeneity ** 0.5
    return heterogeneity


def compute_action_heterogeneity_discrete(actions):
    """Returns score from 0 onwards. As heterogeneity increases, so does the score."""
    action_counts = np.bincount(actions)
    if action_counts.shape[0] < 10:
        action_counts = np.concatenate((action_counts, np.zeros((10 - action_counts.shape[0]))))

    heterogeneity = compute_heterogeneity(action_counts, actions.shape[0])

    #
    # plt.bar(range(10), action_counts)
    # plt.xlabel("Action")
    # plt.ylabel("Frequency")
    # plt.title(f"Heterogeneity: {heterogeneity}")
    # plt.savefig(f"{heterogeneity}.jpg")
    # plt.clf()

    return heterogeneity


def assess_all_dqn(model_name, n=5, buffer_period=200):
    dqn_assay_config_suffixes = ["A", "B", "C", "D", "E", "F", "G", "H"]

    for suffix in dqn_assay_config_suffixes:
        mean_heterogeneity_pre = 0
        mean_heterogeneity_post = 0
        for i in range(1, n+1):
            d = load_data(model_name, f"Interruptions-{suffix}", f"Naturalistic-{i}")

            het_pre = compute_action_heterogeneity_discrete(d["action"][:buffer_period])
            het_post = compute_action_heterogeneity_discrete(d["action"][buffer_period:])
            mean_heterogeneity_pre += het_pre
            mean_heterogeneity_post += het_post
        print("")
        print(suffix)
        print(f"Mean Heterogeneity (pre): {mean_heterogeneity_pre/n}")
        print(f"Mean Heterogeneity (post): {mean_heterogeneity_post/n}")


def assess_all_dqn_binned(model_name, n=5, bin_size=200):
    dqn_assay_config_suffixes = ["A", "B", "C", "D", "E", "F", "G", "H"]
    compiled_results = {}

    for suffix in dqn_assay_config_suffixes:
        bins = []
        score_count_bins = []

        for i in range(1, n+1):
            d = load_data(model_name, f"Interruptions-{suffix}", f"Naturalistic-{i}")
            n_bins_required = math.ceil(d["action"].shape[0]/bin_size)
            while n_bins_required > len(bins):
                bins.append(0)
                score_count_bins.append(0)

            for bin in range(int(n_bins_required)):
                action_slice = d["action"][int(bin*bin_size): int(bin*bin_size)+bin_size]
                bins[bin] += compute_action_heterogeneity_discrete(action_slice)
                score_count_bins[bin] += 1

        bins = np.array(bins) / np.array(score_count_bins)
        compiled_results[suffix] = bins
    return compiled_results


def assess_dqn_binned(model_name, assay_config, n=5, bin_size=200):
    bins = []
    score_count_bins = []

    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"Naturalistic-{i}")
        n_bins_required = math.ceil(d["action"].shape[0]/bin_size)
        while n_bins_required > len(bins):
            bins.append(0)
            score_count_bins.append(0)

        for bin in range(int(n_bins_required)):
            action_slice = d["action"][int(bin*bin_size): int(bin*bin_size)+bin_size]
            bins[bin] += compute_action_heterogeneity_discrete(action_slice)
            score_count_bins[bin] += 1

    bins = np.array(bins) / np.array(score_count_bins)
    compiled_results = bins

    return compiled_results


def assess_all_ppo_binned(model_name, n=5, bin_size=200, mu_i_scaling=10, mu_a_scaling=np.pi/4):
    ppo_assay_config_suffixes = ["V", "W", "X", "Y", "Z"]
    i_compiled_results = {}
    a_compiled_results = {}

    for suffix in ppo_assay_config_suffixes:
        i_bins = []
        a_bins = []
        score_count_bins = []

        for i in range(1, n+1):
            d = load_data(model_name, f"Interruptions-{suffix}", f"Naturalistic-{i}")
            n_bins_required = math.ceil(d["mu_angle"].shape[0]/bin_size)
            while n_bins_required > len(score_count_bins):
                i_bins.append(0)
                a_bins.append(0)
                score_count_bins.append(0)

            for bin in range(int(n_bins_required)):
                impulse_slice = d["mu_impulse"][int(bin*bin_size): int(bin*bin_size)+bin_size, 0] * mu_i_scaling
                angle_slice = d["mu_angle"][int(bin*bin_size): int(bin*bin_size)+bin_size, 0] * mu_a_scaling
                i_het, a_het = compute_action_heterogeneity_continuous(impulse_slice, angle_slice, mu_i_scaling, mu_a_scaling)
                i_bins[bin] += i_het
                a_bins[bin] += a_het
                score_count_bins[bin] += 1

        i_bins = np.array(i_bins) / np.array(score_count_bins)
        a_bins = np.array(a_bins) / np.array(score_count_bins)

        i_compiled_results[suffix] = i_bins
        a_compiled_results[suffix] = a_bins
    return i_compiled_results, a_compiled_results


def assess_ppo_binned(model_name, assay_config, n=5, bin_size=200, mu_i_scaling=10, mu_a_scaling=np.pi/4):
    i_bins = []
    a_bins = []
    score_count_bins = []

    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"Naturalistic-{i}")
        n_bins_required = math.ceil(d["mu_angle"].shape[0]/bin_size)
        while n_bins_required > len(score_count_bins):
            i_bins.append(0)
            a_bins.append(0)
            score_count_bins.append(0)

        for bin in range(int(n_bins_required)):
            impulse_slice = d["mu_impulse"][int(bin*bin_size): int(bin*bin_size)+bin_size, 0] * mu_i_scaling
            angle_slice = d["mu_angle"][int(bin*bin_size): int(bin*bin_size)+bin_size, 0] * mu_a_scaling
            i_het, a_het = compute_action_heterogeneity_continuous(impulse_slice, angle_slice, mu_i_scaling, mu_a_scaling)
            i_bins[bin] += i_het
            a_bins[bin] += a_het
            score_count_bins[bin] += 1

    i_bins = np.array(i_bins) / np.array(score_count_bins)
    a_bins = np.array(a_bins) / np.array(score_count_bins)

    i_compiled_results = i_bins
    a_compiled_results = a_bins
    return i_compiled_results, a_compiled_results


def plot_binned_results(results, model_names, bin_size, initialisation_period, figure_name, control_results=None, indicate_heterogeneity=True):
    configs = results[0].keys()

    fig, axs = plt.subplots(len(configs), sharex=True, figsize=(16, 20))

    colours = ["b", "r", "g", "orange", "k", "black", "yellow", "m"]


    for i, config in enumerate(configs):
        max_score = 0
        heterogeneity_percentages = np.zeros((results[0][config].shape[0]))
        model_counts = np.zeros((results[0][config].shape[0]))

        for model in range(len(results)):
            bin_labels = [i * bin_size for i in range(results[model][config].shape[0])]
            axs[i].plot(bin_labels, results[model][config], color=colours[model])

            if np.max(results[model][config]) > max_score:
                max_score = np.max(results[model][config])

            if indicate_heterogeneity:
                if len(bin_labels) > heterogeneity_percentages.shape[0]:
                    heterogeneity_percentages = np.concatenate((heterogeneity_percentages,
                                                                np.zeros((len(bin_labels)-heterogeneity_percentages.shape[0]))))
                    heterogenous_values = (results[model][config] > 0) * 1

                elif len(bin_labels) < heterogeneity_percentages.shape[0]:
                    heterogenous_values = (np.concatenate((results[model][config],
                                                                  np.zeros((heterogeneity_percentages.shape[0]-len(bin_labels))))) > 0) * 1

                else:
                    heterogenous_values = (results[model][config] > 0) * 1

                if len(bin_labels) > model_counts.shape[0]:
                    model_counts = np.concatenate((model_counts, np.zeros((len(bin_labels)-model_counts.shape[0]))))

                for b, bin in enumerate(bin_labels):
                    model_counts[b] += 1


                heterogeneity_percentages += heterogenous_values

        if indicate_heterogeneity:
            heterogeneity_percentages *= 100/model_counts
            for j, bin in enumerate(heterogeneity_percentages):
                bin = round(bin)
                axs[i].text(j*bin_size, max_score, str(bin) + "%", color="r")

        if control_results is not None:
            for model in range(len(results)):
                bin_labels = [i * bin_size for i in range(control_results[model].shape[0])]
                axs[i].plot(bin_labels, control_results[model], linestyle="dashed", alpha=0.5, color=colours[model])

        axs[i].vlines(initialisation_period, 0, max_score, color="r")
        axs[i].set_ylabel(get_provided_efference(config))


    axs[-1].set_xlabel("Step", fontsize=25)
    fig.text(0.04, 0.5, 'Efference Copy', va='center', ha='center', rotation='vertical', fontsize=25)
    plt.legend(model_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.autoscale()

    plt.savefig(f"{figure_name}.jpg")
    plt.clf()


def display_action_plots_discrete(model_name, assay_config, assay_id, n, figure_name, slice_size=200):
    actions_compiled = []
    plots = [3, 5, 1, 2, 4] + [i for i in range(6, n+1)]  # To place similar ones together.
    for i in plots:
        model_compiled = []
        if i < 6:
            d = load_data(model_name, "Interruptions-HA", f"{assay_id}-{i}")
        else:
            d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        for i in range(0, d["action"].shape[0], slice_size):
            model_compiled.append(d["action"][i: i+slice_size])
        actions_compiled.append(model_compiled)

    # If splitting into bins
    # actions_compiled2 = [[actions_compiled[trial][slice] for trial in range(len(actions_compiled))] for slice in range(10)]
    # actions_compiled2 = [actions_compiled2[i] + [(np.ones(slice_size)*10).astype(int) for j in range(len(actions_compiled2[i]))] for i in range(len(actions_compiled2))]
    actions_compiled2 = actions_compiled
    actions_compiled2 = np.concatenate((actions_compiled2))

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black", "white"]
    color_set = ['b', 'y', 'gold', 'r', 'g', 'lightgreen',  "c", "m", "m", "black", "white"]
    fig, ax = plt.subplots(figsize=(20, 5))
    for i, seq in enumerate(actions_compiled2):
        for j, a in enumerate(seq):
            ax.fill_between((j, j + 1), -i, -i - 1, color=color_set[a])
    plt.vlines(200, 2, -n-2, color="black")

    seen = set()
    seen_add = seen.add
    # actions_compiled_flattened = np.concatenate(np.array(used_sequences))
    actions_present = [i for i in range(10)]
    # actions_present = [x for x in actions_compiled_flattened if not (x in seen or seen_add(x))]
    ordered_actions_present = sorted(actions_present)
    associated_actions = [get_action_name(a) for a in ordered_actions_present]

    legend_elements = [Patch(facecolor=color_set[a], label=associated_actions[i]) for i, a in enumerate(ordered_actions_present)]# [0], [0], marker="o", color=color_set[i], label=associated_actions[i]) for i in actions_present]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.legend(legend_elements, associated_actions, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    # plt.axis("scaled")
    plt.ylabel("Trial", fontsize=30)
    plt.xlabel("Step", fontsize=30)
    plt.savefig(f"{figure_name}.png")
    plt.close(fig)
    plt.clf()


def bin_impulses_and_angles(impulses, angles, max_i, max_a):
    impulse_bins = np.linspace(0, max_i, 11)
    angle_bins = np.linspace(0, max_a, 11)

    discrete_impulses = np.zeros((len(impulses)))
    discrete_angles = np.zeros((len(angles)))

    for i in range(10):
        within_bin = (impulse_bins[i] <= impulses) * (impulses < impulse_bins[i + 1]) * 1
        discrete_impulses[within_bin] = i

        within_bin = (angle_bins[i] <= angles) * (angles < angle_bins[i + 1]) * 1
        discrete_angles[within_bin] = i
    return discrete_impulses, discrete_angles


def display_action_plots_continuous(model_name, assay_config, assay_id, n, figure_name, max_i, max_a, slice_size=200):
    impulses_compiled = []
    angles_compiled = []
    for i in range(1, n+1):
        model_compiled_i = []
        model_compiled_a = []
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        for i in range(0, d["impulse"].shape[0], slice_size):
            model_compiled_i.append(d["impulse"][i: i+slice_size])
            model_compiled_a.append(d["angle"][i: i + slice_size])
        impulses_compiled.append(model_compiled_a)
        angles_compiled.append(model_compiled_a)

    # Bin all
    impulses_compiled, angles_compiled = [[bin_impulses_and_angles(impulses_compiled[i], angles_compiled[i], max_i, max_a)]
                                          for i in range(len(impulses_compiled))]


    actions_compiled2 = [[actions_compiled[trial][slice] for trial in range(len(actions_compiled))] for slice in range(10)]
    actions_compiled2 = [actions_compiled2[i] + [(np.ones(slice_size)*10).astype(int) for j in range(len(actions_compiled2[i]))] for i in range(len(actions_compiled2))]
    actions_compiled2 = np.concatenate((actions_compiled2))

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black", "white"]
    fig, ax = plt.subplots(figsize=(14, 10))
    for i, seq in enumerate(actions_compiled2):
        for j, a in enumerate(reversed(seq)):
            ax.fill_between((j, j + 1), -i, -i - 1, color=color_set[a])

    seen = set()
    seen_add = seen.add
    # actions_compiled_flattened = np.concatenate(np.array(used_sequences))
    actions_present = [i for i in range(10)]
    # actions_present = [x for x in actions_compiled_flattened if not (x in seen or seen_add(x))]
    ordered_actions_present = sorted(actions_present)
    associated_actions = [get_action_name(a) for a in ordered_actions_present]

    legend_elements = [Patch(facecolor=color_set[a], label=associated_actions[i]) for i, a in enumerate(ordered_actions_present)]# [0], [0], marker="o", color=color_set[i], label=associated_actions[i]) for i in actions_present]

    plt.legend(legend_elements, associated_actions, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.axis("scaled")
    plt.savefig(f"{figure_name}.png")
    plt.close(fig)
    plt.clf()


def display_ppo_action_choice(model_name, assay_config, assay_config_control, assay_id, n):
    impulses_compiled = []
    angles_compiled = []

    impulses_compiled_control = []
    angles_compiled_control = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        impulses_compiled.append(data["impulse"])
        angles_compiled.append(data["angle"])

        data = load_data(model_name, assay_config_control, f"{assay_id}-{i}")
        impulses_compiled_control.append(data["impulse"])
        angles_compiled_control.append(data["angle"])

    for impulse, impulse_c in zip(impulses_compiled, impulses_compiled_control):
        plt.plot(impulse, color="blue", alpha=0.5)
        plt.plot(impulse_c, color="orange", alpha=0.5)

    mini, maxi = np.min(np.concatenate((np.concatenate((impulses_compiled)), np.concatenate((impulses_compiled_control))))),\
                 np.max(np.concatenate((np.concatenate((impulses_compiled)), np.concatenate((impulses_compiled_control)))))
    plt.vlines(200, mini, maxi, color="r")
    plt.xlabel("Step")
    plt.ylabel("Impulse")
    plt.savefig("PPO impulses.png")
    plt.clf()

    for angle, angle_c in zip(angles_compiled, angles_compiled_control):
        plt.plot(angle, color="blue", alpha=0.5)
        plt.plot(angle_c, color="orange", alpha=0.5)

    mina, maxa = np.min(np.concatenate((np.concatenate((angles_compiled)), np.concatenate((angles_compiled_control))))),\
                 np.max(np.concatenate((np.concatenate((angles_compiled)), np.concatenate((angles_compiled_control)))))
    plt.vlines(200, mina, maxa,  color="r")
    plt.xlabel("Step")
    plt.ylabel("Angle (pi radians)")
    plt.savefig("PPO angles.png")
    plt.clf()


if __name__ == "__main__":

    # display_ppo_action_choice("ppo_scaffold_21-2", "Interruptions-Z", "Behavioural-Data-Free", "Naturalistic", 5)
    # display_action_plots_continuous("ppo_scaffold_21-2", "Interruptions-W", "Naturalistic", 5, "ppo_21_2_actions",
    #                                 max_i=16, max_a=1)
    # PPO
    # ppo_control_i, ppo_control_a = assess_ppo_binned("ppo_scaffold_21-1", "Behavioural-Data-Free", 5, 101, mu_i_scaling=16, mu_a_scaling=1)
    # ppo_control_i2, ppo_control_a2 = assess_ppo_binned("ppo_scaffold_21-2", "Behavioural-Data-Free", 5, 101, mu_i_scaling=16, mu_a_scaling=1)
    #
    # ppo_results_i, ppo_results_a = assess_all_ppo_binned("ppo_scaffold_21-1", 5, 101, mu_i_scaling=16, mu_a_scaling=1)
    # ppo_results_i2, ppo_results_a2 = assess_all_ppo_binned("ppo_scaffold_21-2", 5, 101, mu_i_scaling=16, mu_a_scaling=1)
    #
    # plot_binned_results([ppo_results_i, ppo_results_i2], ["ppo_scaffold_21-1", "ppo_scaffold_21-2"], 101,
    #                     200, figure_name="all_ppo_heterogeneity_impulse", control_results=[ppo_control_i, ppo_control_i2])
    # plot_binned_results([ppo_results_a, ppo_results_a2], ["ppo_scaffold_21-1", "ppo_scaffold_21-2"], 101,
    #                     200, figure_name="all_ppo_heterogeneity_angle", control_results=[ppo_control_a, ppo_control_a2])


    # DQN
    display_action_plots_discrete("dqn_scaffold_14-1", "Interruptions-H", "Naturalistic", 35,
                                  figure_name="dqn_14_1_H_actions", slice_size=2000)
    # display_action_plots_discrete("dqn_scaffold_30-2", "Interruptions-H", "Naturalistic", 5,
    #                               figure_name="dqn_30_2_H_actions", slice_size=100)
    # display_action_plots_discrete("dqn_scaffold_18-2", "Interruptions-H", "Naturalistic", 5,
    #                               figure_name="dqn_18_2_H_actions", slice_size=100)

    # results1 = assess_all_dqn_binned("dqn_scaffold_18-1", 5, 101)
    # results2 = assess_all_dqn_binned("dqn_scaffold_18-2", 5, 101)
    # results3 = assess_all_dqn_binned("dqn_scaffold_14-1", 5, 101)
    # results4 = assess_all_dqn_binned("dqn_scaffold_14-2", 5, 101)
    # results5 = assess_all_dqn_binned("dqn_scaffold_26-1", 5, 101)
    # results6 = assess_all_dqn_binned("dqn_scaffold_26-2", 5, 101)
    # results7 = assess_all_dqn_binned("dqn_scaffold_30-1", 5, 101)
    # results8 = assess_all_dqn_binned("dqn_scaffold_30-2", 5, 101)
    #
    # control1 = assess_dqn_binned("dqn_scaffold_18-1", "Behavioural-Data-Free", 10, 101)
    # control2 = assess_dqn_binned("dqn_scaffold_18-2", "Behavioural-Data-Free", 10, 101)
    # control3 = assess_dqn_binned("dqn_scaffold_14-1", "Behavioural-Data-Free", 10, 101)
    # control4 = assess_dqn_binned("dqn_scaffold_14-2", "Behavioural-Data-Free", 10, 101)
    # control5 = assess_dqn_binned("dqn_scaffold_18-1", "Behavioural-Data-Free", 10, 101)
    # control6 = assess_dqn_binned("dqn_scaffold_18-2", "Behavioural-Data-Free", 10, 101)
    # control7 = assess_dqn_binned("dqn_scaffold_14-1", "Behavioural-Data-Free", 10, 101)
    # control8 = assess_dqn_binned("dqn_scaffold_14-2", "Behavioural-Data-Free", 10, 101)
    #
    # model_names = ["dqn_scaffold_18-1", "dqn_scaffold_18-2", "dqn_scaffold_14-1", "dqn_scaffold_14-2",
    #                "dqn_scaffold_26-1", "dqn_scaffold_26-2", "dqn_scaffold_30-1", "dqn_scaffold_30-2"]
    #
    # plot_binned_results([results1, results2, results3, results4, results5, results6, results7, results8],
    #                     model_names, bin_size=101, initialisation_period=200, figure_name="all_dqn_heterogeneity",
    #                     control_results=[control1, control2, control3, control4, control5, control6, control7, control8])


    # Example for nearly fully heterogenous actions
    # nearly_uniform = np.repeat(np.array([i for i in range(10)]), 8, 0)
    # nearly_uniform[0] = 1
    # compute_action_heterogeneity_discrete(nearly_uniform)

