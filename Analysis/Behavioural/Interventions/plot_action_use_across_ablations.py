import matplotlib.pyplot as plt
import seaborn as sns


from Analysis.Behavioural.New.show_spatial_density import get_action_name
from Analysis.load_data import load_data


def get_action_freq(model, ablation_config, per, number_of_trials):
    actions = [0 for i in range(10)]
    for i in range(1, number_of_trials+1):
        data = load_data(model, f"Ablation-{ablation_config}", f"Ablated-{per}-{i}")
        for a in range(10):
            actions[a] += len([ac for ac in data["behavioural choice"] if ac == a])
    return actions


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

    plt.xlabel("Percentage Ablated")
    plt.ylabel("Action Counts")
    plt.show()


plot_action_use("new_even_prey_ref-1", "Indiscriminate-even_prey_only", 3, range(0, 105, 5))
plot_action_use("new_even_prey_ref-1", "Test-Prey-Large-Central-even_prey_only", 3)
plot_action_use("new_even_prey_ref-4", "Test-Predator_Only-even_prey_only", 3)

