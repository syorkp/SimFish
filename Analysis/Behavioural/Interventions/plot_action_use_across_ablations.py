import matplotlib.pyplot as plt
import seaborn as sns


from Analysis.Behavioural.New.show_spatial_density import get_action_name
from Analysis.load_data import load_data
import numpy as np


def get_action_freq(model, ablation_config, per, number_of_trials):
    actions = [0 for i in range(10)]
    for i in range(1, number_of_trials+1):
        data = load_data(model, f"Ablation-{ablation_config}", f"Ablated-{per}-{i}")
        for a in range(10):
            actions[a] += len([ac for ac in data["behavioural choice"] if ac == a])
    return actions


def get_action_freq_predator_sequences(model, ablation_config, per, number_of_trials):
    actions = [0 for i in range(10)]
    for i in range(1, number_of_trials+1):
        data = load_data(model, f"Ablation-{ablation_config}", f"Ablated-{per}-{i}")
        for a in range(10):
            predator_timestamps = [i-1 for i in data["step"] if data["predator"][i-1] == 1]
            actions[a] += len([ac for i, ac in enumerate(data["behavioural choice"]) if ac == a and i in predator_timestamps])
    return actions


def plot_action_use_multiple_models(models, ablation_config, number_of_trials, ablation_groups=range(0, 105, 10)):
    sns.set()
    action_use = [[[] for m in models] for i in range(10)]
    for per in ablation_groups:
        if per == 35: per=40
        for m, model in enumerate(models):
            actions = get_action_freq(model, ablation_config, per, number_of_trials)
            for i, a in enumerate(actions):
                action_use[i][m].append(a)


    high_v= [[] for i in range(10)]
    low_v = [[] for i in range(10)]

    action_use = np.array(action_use)
    average_action_use = [[] for i in range(10)]
    for a, action in enumerate(average_action_use):
        for p, per in enumerate(ablation_groups):
            high_v[a].append(max(action_use[a, :, p]))
            low_v[a].append(min(action_use[a, :, p]))
            average_action_use[a].append(np.mean(action_use[a, :, p]))

    fig, ax = plt.subplots(figsize=(7, 7))
    for a, action in enumerate(average_action_use):
        ax.plot(ablation_groups, action, label=get_action_name(a))
        low_v_x = low_v[a]
        high_v_x = high_v[a]
        ax.fill_between(ablation_groups, low_v_x, high_v_x, alpha=0.2)

    ax.legend()
    plt.xlabel("Percentage ablated", fontsize=15)
    plt.ylabel("Bout Counts", fontsize=15)
    plt.show()


def plot_action_use(model, ablation_config, number_of_trials, ablation_groups=range(0, 105, 10)):
    sns.set()
    action_use = [[] for i in range(10)]
    for per in ablation_groups:
        if per == 35: per=40
        actions = get_action_freq(model, ablation_config, per, number_of_trials)
        for i, a in enumerate(actions):
            action_use[i].append(a)

    fig, ax = plt.subplots(figsize=(7, 7))
    for a, action in enumerate(action_use):
        ax.plot(ablation_groups, action, label=get_action_name(a))

    ax.legend()

    plt.xlabel("Percentage ablated", fontsize=15)
    plt.ylabel("Bout Counts", fontsize=15)
    plt.show()


def plot_action_use_multiple_models_predator_sequences(models, ablation_config, number_of_trials, ablation_groups=range(0, 105, 10)):
    sns.set()
    action_use = [[[] for m in models] for i in range(10)]
    for per in ablation_groups:
        if per == 35: per = 40
        for m, model in enumerate(models):
            actions = get_action_freq_predator_sequences(model, ablation_config, per, number_of_trials)
            for i, a in enumerate(actions):
                action_use[i][m].append(a)

    high_v = [[] for i in range(10)]
    low_v = [[] for i in range(10)]

    action_use = np.array(action_use)
    average_action_use = [[] for i in range(10)]
    for a, action in enumerate(average_action_use):
        for p, per in enumerate(ablation_groups):
            high_v[a].append(max(action_use[a, :, p]))
            low_v[a].append(min(action_use[a, :, p]))
            average_action_use[a].append(np.mean(action_use[a, :, p]))

    fig, ax = plt.subplots(figsize=(7, 7))
    for a, action in enumerate(average_action_use):
        ax.plot(ablation_groups, action, label=get_action_name(a))
        low_v_x = low_v[a]
        high_v_x = high_v[a]
        ax.fill_between(ablation_groups, low_v_x, high_v_x, alpha=0.2)

    ax.legend()

    plt.xlabel("Percentage of neurons ablated", fontsize=15)
    plt.ylabel("Bout Counts", fontsize=15)
    plt.show()


# plot_action_use_multiple_models_predator_sequences(["new_even_prey_ref-4", "new_even_prey_ref-5", "new_even_prey_ref-8"], "Indiscriminate-even_predator", 6, range(0, 105, 5))
plot_action_use_multiple_models(["new_even_prey_ref-1", "new_even_prey_ref-2", "new_even_prey_ref-3", "new_even_prey_ref-4"], "Indiscriminate-even_prey_only", 3, range(0, 105, 5))
# plot_action_use("new_even_prey_ref-1", "Test-Prey-Large-Central-even_prey_only", 3)
# plot_action_use("new_even_prey_ref-4", "Test-Predator_Only-even_prey_only", 3)

# plot_action_use_multiple_models(["new_even_prey_ref-1", "new_even_prey_ref-2", "new_even_prey_ref-2", "new_even_prey_ref-4"], "Ablation-Test-Prey-Large-Central-even_prey_only", 3, range(0, 105, 5))
# plot_action_use_multiple_models(["new_even_prey_ref-1", "new_even_prey_ref-2", "new_even_prey_ref-3", "new_even_prey_ref-4"], "Test-Prey-Large-Central-even_prey_only", 3, range(0, 105, 10))