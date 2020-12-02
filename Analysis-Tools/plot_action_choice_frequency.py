import csv

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def clean_action_choice_data(reader, file_name):
    """For Action choice file located given the location of, reads, and ensures that no subsequent entry is of a lower
    step value than any other, increasing their step value when this is the case. Should be unnecessary in future"""
    increment = 0
    previous_step = 0
    with open(file_name, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["Wall time", "Step", "Value"])
        for i, row in enumerate(reader):
            if i == 0:
                pass
            else:
                current_step = int(row[1])
                current_step += increment
                if current_step <= previous_step:
                    current_step += (previous_step-increment)
                    increment += (previous_step-increment)
                writer.writerow([row[0], current_step, row[2]])
                previous_step = current_step


def get_action_choice(model, action):
    with open(f"{model}/run-logs-tag-action {action}", "r") as original_file:
        csv_reader = csv.reader(original_file, delimiter=',')
        x = []
        y = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                pass
            else:
                x.append(int(row[1]))
                y.append(float(row[2]))
    return x, y


def compute_running_averages(x_data, y_data):
    window_size = 20
    i = 0
    moving_averages = []
    while i < len(y_data) - window_size + 1:
        this_window = y_data[i: i + window_size]
        window_average = sum(this_window)/window_size
        moving_averages.append(window_average)
        i += 1
    # while len(moving_averages) < len(y_data):
    #     moving_averages.append(0)
    while len(moving_averages) < len(x_data):
        del x_data[-1]

    return x_data, moving_averages


def cut_action_data(x_data, y_data, length):
    for i, value in enumerate(x_data):
        if value > length:
            del x_data[i:]
            del y_data[i:]
            break
    return x_data, y_data


def create_action_plots(action):
    # Get the data
    base_1_x, base_1_y = get_action_choice("base_1", str(action))
    base_2_x, base_2_y = get_action_choice("base_2", str(action))
    base_3_x, base_3_y = get_action_choice("base_3", str(action))
    # new_test_1_x, new_test_1_y = get_action_choice("new_test_1", str(action))

    # Compute running averages
    base_1_x, base_1_y = compute_running_averages(base_1_x, base_1_y)
    base_2_x, base_2_y = compute_running_averages(base_2_x, base_2_y)
    base_3_x, base_3_y = compute_running_averages(base_3_x, base_3_y)
    # new_test_1_x, new_test_1_y = compute_running_averages(new_test_1_x, new_test_1_y)

    # Cut to 2 million steps
    base_1_x, base_1_y = cut_action_data(base_1_x, base_1_y, 2002000)
    base_2_x, base_2_y = cut_action_data(base_2_x, base_2_y, 2002000)
    base_3_x, base_3_y = cut_action_data(base_3_x, base_3_y, 2002000)
    # new_test_1_x, new_test_1_y = cut_action_data(new_test_1_x, new_test_1_y, 2002000)

    plt.plot(base_1_x, base_1_y, "b")
    plt.plot(base_2_x, base_2_y, "g")
    plt.plot(base_3_x, base_3_y, "r")
    plt.tick_params(labelsize=10)

    # plt.plot(new_test_1_x, new_test_1_y, "y")
    plt.title(f"Frequency of action {str(action)} over time", fontsize=15)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Action frequency", fontsize=12)
    plt.show()


def create_boxplots(index, models, timestamp):
    # TODO: Note is not accurate at present - dont select on index, select on y-value proximity.
    # TODO: FIx this for final data.
    data = {"action": [], "model": [], "frequency": []}
    action_lists = []
    for action in range(7):
        # points = []
        for model in models:
            data["action"].append(action)
            data["model"].append(model)
            x, y = get_action_choice(model, action)
            data["frequency"].append(y[index])

        # action_lists.append(points)
    # data["values"] = action_lists
    data = pd.DataFrame(data)
    ax = sns.boxplot(y="frequency", x="action", data=data, whis=np.inf)
    ax = sns.stripplot(y="frequency", x="action", data=data, color=".3")
    ax.tick_params(labelsize=10)
    plt.title(f"Frequency of actions at step {str(timestamp)}")
    plt.xlabel("Action", fontsize=12)
    plt.ylabel("Action frequency", fontsize=12)
    plt.show()


# Clean action choice data found in downloads, then save to new files.
filenames = ["run-logs-tag-action 0",
             "run-logs-tag-action 1",
             "run-logs-tag-action 2",
             "run-logs-tag-action 3",
             "run-logs-tag-action 4",
             "run-logs-tag-action 5",
             "run-logs-tag-action 6",
             ]

for i in filenames:
    with open(f"/home/sam/Downloads/{i}.csv") as original_file:
        csv_reader = csv.reader(original_file, delimiter=',')
        clean_action_choice_data(csv_reader, i)

# Produce box plots for action choice frequency
models = ["base_1", "base_2", "base_3", "new_test_1"]
create_boxplots(500, models, 1000000)
create_boxplots(999, models, 2000000)

# Produce running average action frequency graphs
create_action_plots(0)
create_action_plots(1)
create_action_plots(6)




