import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from load_data import load_data


def plot_activity(rnn_data):
    fig, axs = plt.subplots(4, 1, sharex=True)

    axs[0].plot(rnn_data[0])
    axs[0].set_ylabel("Unit 1 activity", fontsize=25)
    axs[1].plot(rnn_data[1])
    axs[1].set_ylabel("Unit 2 activity", fontsize=25)
    axs[2].plot(rnn_data[2])
    axs[2].set_ylabel("Unit 3 activity", fontsize=25)
    axs[3].plot(rnn_data[3])
    axs[3].set_ylabel("Unit 4 activity", fontsize=25)
    axs[3].set_xlabel("Step", fontsize=25)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[2].tick_params(labelsize=15)
    axs[3].tick_params(labelsize=15)

    # axs[0].set_ylim(0.5, 1.5)
    fig.set_size_inches(18.5, 20)
    fig.savefig('test2png.png', dpi=100)
    plt.show()

data = load_data("Prey Stimuli", "Visual-Stimulus-Assay-2")


rnn_unit_1 = [i["rnn state"][0][0] for i in data]
rnn_unit_2 = [i["rnn state"][0][1] for i in data]
rnn_unit_3 = [i["rnn state"][0][2] for i in data]
rnn_unit_200 = [i["rnn state"][0][199] for i in data]

unit_activity = [rnn_unit_1, rnn_unit_2, rnn_unit_3, rnn_unit_200]
plot_activity(unit_activity)
