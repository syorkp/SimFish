import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from Analysis.load_data import load_data


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


data = load_data("changed_penalties-2", "Naturalistic", "Naturalistic-1")


rnn_unit_1 = [data["rnn state"][i-1][0][0] for i in data["step"]]
rnn_unit_3 = [data["rnn state"][i-1][0][19] for i in data["step"]]
rnn_unit_200 = [data["rnn state"][i-1][0][190] for i in data["step"]]
conv_unit_1 = [data["left_conv_1"][i-1][0][0] for i in data["step"]]

unit_activity = [rnn_unit_1, rnn_unit_3, rnn_unit_200, conv_unit_1]
plot_activity(unit_activity)
