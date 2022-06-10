import matplotlib.pyplot as plt
import numpy as np
import seaborn

from Analysis.load_data import load_data


def create_conv_layer_diagram(width, height):
    fig, ax = plt.subplots()
    layer = plt.Polygon(np.array([[0, 0], [0, height], [width, height], [width, 0]]))
    ax.add_patch(layer)
    plt.axis("scaled")
    plt.axis('off')

    plt.show()


def create_neuron_layer_diagram(width, n_units, recurrent=False):
    area_available = width/n_units
    height = area_available

    fig, ax = plt.subplots()
    layer = plt.Polygon(np.array([[0, 0], [0, height], [width, height], [width, 0]]), fc="white", ec="red")
    ax.add_patch(layer)

    circle_radius = area_available * 0.4
    for i in range(n_units):
        if recurrent:
            neuron = plt.Polygon(np.array([[(i*area_available)+(area_available/5), height/5],
                                           [(i * area_available) + (area_available / 5), 4 * height / 5],
                                           [(i * area_available) + (4 * area_available / 5), 4 * height / 5],
                                           [(i * area_available) + (4 * area_available / 5), height / 5],
                                           ]), ec="blue")
        else:
            neuron = plt.Circle(((i*area_available)+(area_available/2), height/2), circle_radius, ec="blue")
        ax.add_patch(neuron)

    plt.axis("scaled")
    plt.axis('off')

    plt.show()


def show_energy_state(data, datapoint_index):
    plt.plot(data["energy_state"])

    et = data["energy_state"][datapoint_index]
    plt.hlines(et, 0, datapoint_index, colors="r")
    plt.vlines(datapoint_index, 0, et, colors="r")

    plt.margins(x=0, y=0)
    plt.show()


create_neuron_layer_diagram(100, 10, True)
dat = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
show_energy_state(dat, 1000)