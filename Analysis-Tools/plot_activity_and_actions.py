import json
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

with open("../Assay-Output/base-1/Visual-Stimulus-Assay-1.json", "r") as file:
    data = json.load(file)

# TODO: Create code to load everything into a dataframe - should be own function for loading data.


# Shorten action choice

# action_choice = action_choice[200: 500]
# rnn_unit_1 = rnn_unit_1[200: 500]
# rnn_unit_2 = rnn_unit_2[200: 500]
# rnn_unit_3 = rnn_unit_3[200: 500]
# rnn_unit_200 = rnn_unit_200[200: 500]


def calculate_average_rnn_activity():
    averages = []
    for i, step in enumerate(data):
        total_rnn_activity = 0
        for unit in range(len(data[0]["rnn state"][0])):
            total_rnn_activity += data[i]["rnn state"][0][unit]
        average = total_rnn_activity / len(data[0]["rnn state"][0])
        averages.append(average)
    return averages


def plot_activity_and_action(rnn_data, action_number, action_data):
    action_timestamps = [i for i, a in enumerate(action_data) if a == action_number]
    fig, axs = plt.subplots(2, 1, sharex=True)

    axs[0].eventplot(action_timestamps)
    axs[1].plot(rnn_data)
    axs[0].set_ylim(0.5, 1.5)
    plt.show()


def plot_activity_and_actions(rnn_data, action_data):
    # colour_sequence = []
    # TODO: Final three not right.
    # for action in action_data:
    #     if action == 0:
    #         colour_sequence.append((0.0, 0.0, 1.0))
    #     elif action == 1:
    #         colour_sequence.append((0.75, 0.75, 0.0))
    #     elif action == 2:
    #         colour_sequence.append((0.0, 0.75, 0.75))
    #     elif action == 3:
    #         colour_sequence.append((1.0, 0.0, 0.0))
    #     elif action == 4:
    #         colour_sequence.append((0.75, 0.0, 0.75))
    #     elif action == 5:
    #         colour_sequence.append((0.75, 0.0, 0.75))
    #     elif action == 6:
    #         colour_sequence.append((0.75, 0.0, 0.75))
    #     else:
    #         print("error")
    separated_actions = []
    for action in range(7):
        action_timestamps = [i for i, a in enumerate(action_data) if a == action]
        if len(action_timestamps) > 0:
            separated_actions.append(action_timestamps)
    # colorCodes = np.array([(0.0, 0.0, 1.0),
    #                        (0.0, 0.75, 0.75),
    #                        (0.75, 0.75, 0.0),
    #                        (1.0, 0.0, 0.0),
    #                        (0.75, 0.0, 0.75),
    #                        (0.75, 0.0, 0.75),
    #                        (0.75, 0.0, 0.75)])
    colorCodes = np.array([(0.0, 0.0, 1.0),
                           (0.0, 0.75, 0.75),
                           (0.75, 0.75, 0.0),
                           (1.0, 0.0, 0.0),])
    fig, axs = plt.subplots(4, 1, sharex=True)
    # positions = range(len(colour_sequence))
    # colour_sequence = colour_sequence[1:]
    # plt.eventplot(separated_actions, color=colorCodes)

    axs[0].eventplot(separated_actions, color=colorCodes)
    axs[0].set_ylabel("Action Choice", fontsize=25)
    axs[1].plot(rnn_data[0])
    axs[1].set_ylabel("Unit 1 activity", fontsize=25)
    axs[2].plot(rnn_data[1])
    axs[2].set_ylabel("Unit 2 activity", fontsize=25)
    axs[3].plot(rnn_data[2])
    axs[3].set_ylabel("Unit 3 activity", fontsize=25)
    # axs[4].plot(rnn_data[3])
    # axs[4].set_ylabel("Unit 4 activity", fontsize=25)
    axs[3].set_xlabel("Step", fontsize=25)
    axs[0].tick_params(labelsize=15)
    axs[1].tick_params(labelsize=15)
    axs[2].tick_params(labelsize=15)
    axs[3].tick_params(labelsize=15)
    # axs[4].tick_params(labelsize=15)

    # axs[0].set_ylim(0.5, 1.5)
    fig.set_size_inches(18.5, 20)
    fig.savefig('test2png.png', dpi=100)
    plt.show()


# To print individual for each action
# for i in range(6):
#     plot_activity_and_action(rnn_unit_1, i, action_choice)
#     plot_activity_and_action(rnn_unit_2, i, action_choice)
#     plot_activity_and_action(rnn_unit_3, i, action_choice)
#     plot_activity_and_action(rnn_unit_200, i, action_choice)

# For all actions
rnn_unit_1 = [i["rnn state"][0][0] for i in data]
rnn_unit_3 = [i["rnn state"][0][19] for i in data]
rnn_unit_200 = [i["rnn state"][0][190] for i in data]

action_choice = [i["behavioural choice"] for i in data]
unit_activity = [rnn_unit_1, rnn_unit_3, rnn_unit_200]
plot_activity_and_actions(unit_activity, action_choice)

# plt.plot(calculate_average_rnn_activity())
# plt.show()
