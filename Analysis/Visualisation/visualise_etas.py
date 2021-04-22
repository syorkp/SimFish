import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Neural.event_triggered_averages import get_eta
from Analysis.Neural.calculate_vrv import normalise_vrvs
from Analysis.Visualisation.visualise_response_vectors import order_vectors_by_kmeans
from Analysis.Behavioural.show_spatial_density import get_action_name


def display_all_atas(atas):
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 20)
    atas = [atas[key] for key in atas.keys()]
    atas = [[atas[action][neuron] for action in range(len(atas))] for neuron in range(len(atas[0]))]
    for i, neuron in enumerate(atas):
        for j, action in enumerate(neuron):
            if action >1000:
                atas[i][j] = 1000
    # atas = sorted(atas, key=lambda x: x[3])
    atas = normalise_vrvs(atas)
    atas = order_vectors_by_kmeans(atas)
    ax.pcolor(atas, cmap='coolwarm')
    # ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    plt.xticks(range(10), ["                    " + get_action_name(i) for i in range(10)], fontsize=15)
    # ax.set_yticks(lat_grid, minor=True)

    plt.show()


data = load_data("even_prey_ref-3", "Behavioural-Data-Free", "Naturalistic-1")
ata = get_eta(data, "actions")
display_all_atas(ata)