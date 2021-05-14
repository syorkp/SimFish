from matplotlib import pyplot as plt
from Analysis.load_data import load_data


def display_consumption_and_predator_times(data):

    consumption_timestamps = [i for i, a in enumerate(data["consumed"]) if a == 1]
    predator_timestamps = [i for i, a in enumerate(data["predator"]) if a == 1]

    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].eventplot(consumption_timestamps)
    axs[1].eventplot(predator_timestamps)
    axs[0].set_ylabel("Prey Consumed")
    axs[1].set_ylabel("Predators Present")
    plt.show()


data = load_data("changed_penalties-2", "Naturalistic", "Naturalistic-1")
display_consumption_and_predator_times(data)

