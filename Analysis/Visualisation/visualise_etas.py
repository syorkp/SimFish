import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Neural.event_triggered_averages import get_eta


def display_all_atas(atas):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 20)
    atas = [atas[key] for key in atas.keys()]
    atas = [[abs(atas[action][neuron]) for action in range(len(atas))] for neuron in range(len(atas[0]))]
    for i, neuron in enumerate(atas):
        for j, action in enumerate(neuron):
            if action >1000:
                atas[i][j] = 1000
    atas = sorted(atas, key=lambda x: x[3])
    ax.pcolor(atas, cmap='Reds')
    # ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    ax.set_xticks(range(0, 11), minor=True)
    # ax.set_yticks(lat_grid, minor=True)
    plt.show()


data = load_data("even_prey_ref-4", "Naturalistic", "Naturalistic-1")
ata = get_eta(data, "actions")
display_all_atas(ata)