import matplotlib.pyplot as plt
import numpy as np
import math

from Analysis.load_data import load_data


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


def plot_binned_results(results, model_names, bin_size, initialisation_period):
    configs = results[0].keys()

    fig, axs = plt.subplots(len(configs), sharex=True)

    for i, config in enumerate(configs):
        max_score = 0

        for model in range(len(results)):
            bin_labels = [i * bin_size for i in range(results[model][config].shape[0])]
            axs[i].plot(bin_labels, results[model][config])

            if np.max(results[model][config]) > max_score:
                max_score = np.max(results[model][config])

        axs[i].vlines(initialisation_period, 0, max_score, color="r")
        axs[i].set_ylabel(get_provided_efference(config))
    axs[-1].set_xlabel("Step")
    fig.text(0.04, 0.5, 'Efference Copy', va='center', ha='center', rotation='vertical')#, fontsize=rcParams['axes.labelsize'])
    plt.legend(model_names, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.autoscale()

    plt.show()


if __name__ == "__main__":
    actions = load_data("ppo_scaffold_21-2", "Interruptions-W", "Naturalistic-4")
    ppo_control_i, ppo_control_a = assess_ppo_binned("ppo_scaffold_21-2", "Behavioural-Data-Free", 5, 101, mu_i_scaling=16, mu_a_scaling=1)

    ppo_results_i, ppo_results_a = assess_all_ppo_binned("ppo_scaffold_21-1", 5, 101, mu_i_scaling=16, mu_a_scaling=1)
    ppo_results_i2, ppo_results_a2 = assess_all_ppo_binned("ppo_scaffold_21-2", 5, 101, mu_i_scaling=16, mu_a_scaling=1)

    ppo_control_i = {key: ppo_control_i for key in ppo_results_i.keys()}
    ppo_control_a = {key: ppo_control_a for key in ppo_results_a.keys()}


    plot_binned_results([ppo_results_i, ppo_results_i2, ppo_control_i], ["ppo_scaffold_21-1", "ppo_scaffold_21-2", "Control"], 101, 200)
    plot_binned_results([ppo_results_a, ppo_results_a2, ppo_control_a], ["ppo_scaffold_21-1", "ppo_scaffold_21-2", "Control"], 101, 200)
    # nearly_uniform = np.repeat(np.array([i for i in range(10)]), 8, 0)
    # nearly_uniform[0] = 1
    # compute_action_heterogeneity_discrete(nearly_uniform)

    results1 = assess_all_dqn_binned("dqn_scaffold_18-1", 5, 101)
    results2 = assess_all_dqn_binned("dqn_scaffold_18-2", 5, 101)
    results3 = assess_all_dqn_binned("dqn_scaffold_14-1", 5, 101)
    results4 = assess_all_dqn_binned("dqn_scaffold_14-2", 5, 101)
    results5 = assess_all_dqn_binned("dqn_scaffold_26-1", 5, 101)
    results6 = assess_all_dqn_binned("dqn_scaffold_26-2", 5, 101)
    # results7 = assess_all_dqn_binned("dqn_scaffold_30-1", 5, 80)
    # results8 = assess_all_dqn_binned("dqn_scaffold_30-2", 5, 80)

    actions = load_data("dqn_scaffold_14-1", "Interruptions-H", "Naturalistic-1")

    model_names = ["dqn_scaffold_18-1", "dqn_scaffold_18-2", "dqn_scaffold_14-1", "dqn_scaffold_14-2", "dqn_scaffold_26-1", "dqn_scaffold_26-2"]
    # results4 = assess_all_dqn_binned("dqn_scaffold_14-2", 5, 80)
    plot_binned_results([results1, results2, results3, results4, results5, results6], model_names, bin_size=101, initialisation_period=200, )
    # assess_all_dqn("dqn_scaffold_18-2", 5, 200)



