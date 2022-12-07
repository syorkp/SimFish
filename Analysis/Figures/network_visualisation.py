import matplotlib.pyplot as plt
import numpy as np
import seaborn

from Analysis.load_data import load_data


def create_conv_layer_diagram(width, height):
    fig, ax = plt.subplots()
    layer = plt.Polygon(np.array([[0, 0], [0, height], [width, height], [width, 0]]), hatch="/")
    ax.add_patch(layer)
    plt.axis("scaled")
    plt.axis('off')
    plt.savefig(f"./Panels/Panel-2/conv-{width}-{height}.jpg")
    plt.show()


def create_neuron_layer_diagram(width, n_units, recurrent=False):
    area_available = width/n_units
    height = area_available

    fig, ax = plt.subplots(figsize=(20, 2))
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
    plt.savefig(f"./Panels/Panel-2/neuron-recurrent={recurrent}.jpg")

    plt.show()


create_conv_layer_diagram(80, 40)
create_conv_layer_diagram(40, 20)
create_conv_layer_diagram(40, 10)
create_conv_layer_diagram(100, 10)
create_neuron_layer_diagram(300, 20, False)
create_neuron_layer_diagram(300, 20, True)
