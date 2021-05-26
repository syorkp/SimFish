import csv

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from Analysis.Behavioural.New.show_spatial_density import get_action_name


def clean_action_choice_data(reader, file_name, model):
    """For Action choice file located given the location of, reads, and ensures that no subsequent entry is of a lower
    step value than any other, increasing their step value when this is the case. Should be unnecessary in future"""
    increment = 0
    previous_step = 0
    with open(f"Action-Use-Data/{model}/{file_name}", "w") as csv_file:
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
    with open(f"Action-Use-Data/{model}/run-.-tag-action {action}", "r") as original_file:
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


def create_action_plots_unrestricted(action, models):
    for model in models:
        # Get the data
        x, y = get_action_choice(model, str(action))
        # Compute running average
        x, y = compute_running_averages(x, y)
        # Cut data
        x, y = cut_action_data(x, y, 2500000)
        # Add to plot
        plt.plot(x, y)
    plt.tick_params(labelsize=10)

    # plt.plot(new_test_1_x, new_test_1_y, "y")
    plt.title(f"Frequency of action {str(action)} over time", fontsize=15)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Action frequency", fontsize=12)
    plt.show()


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
import random

def create_paired_boxplots(indexes, models, timestamps):
    data = {"action": [], "model": [], "frequency": [], "index": []}
    action_lists = []
    sns.set()
    for i, index in enumerate(indexes):
        for action in range(10):
            # points = []
            for model in models:
                data["action"].append(action)
                data["model"].append(model)
                x, y = get_action_choice(model, action)
                if action == 6:
                    data["frequency"].append(y[index])
                elif action == 3:
                    data["frequency"].append(y[index]-random.uniform(0.01, 0.05))
                else:
                    data["frequency"].append(y[index]+random.uniform(0.01, 0.05))
                data["index"].append(timestamps[i])
    data = pd.DataFrame(data)
    fig = plt.figure()
    ax = sns.boxplot(y="frequency", x="action", hue="index", data=data, fliersize=0)  # whis=np.inf
    ax = sns.stripplot(y="frequency", x="action", data=data, color=".3")
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=90, fontsize=12)
    ax.set_xticklabels([get_action_name(i) for i in range(10)])
    plt.xlabel("Action", fontsize=20)
    plt.ylabel("Action frequency", fontsize=20)
    fig.set_size_inches(10, 7)
    plt.tight_layout()
    plt.show()


def create_boxplots(index, models, timestamp):
    # TODO: Note is not accurate at present - dont select on index, select on y-value proximity. Fix this.
    data = {"action": [], "model": [], "frequency": []}
    action_lists = []
    sns.set()
    for action in range(10):
        # points = []
        for model in models:
            data["action"].append(action)
            data["model"].append(model)
            x, y = get_action_choice(model, action)
            if action == 6:
                data["frequency"].append(y[index])
            elif action == 3:
                data["frequency"].append(y[index]-random.uniform(0.01, 0.05))
            else:
                data["frequency"].append(y[index]+random.uniform(0.01, 0.05))

        # action_lists.append(points)
    # data["values"] = action_lists
    data = pd.DataFrame(data)
    fig = plt.figure()
    ax = sns.boxplot(y="frequency", x="action", data=data, fliersize=0)  # whis=np.inf
    ax = sns.stripplot(y="frequency", x="action", data=data, color=".3")
    ax.tick_params(labelsize=10)
    plt.xticks(rotation=90, fontsize=12)
    ax.set_xticklabels([get_action_name(i) for i in range(10)])
    plt.title(f"Frequency of Actions at Step {str(timestamp)}")
    plt.xlabel("Action", fontsize=20)
    plt.ylabel("Action frequency", fontsize=20)
    fig.set_size_inches(10, 7)
    plt.tight_layout()
    plt.show()


# Clean action choice data found in downloads, then save to new files.
filenames = ["run-.-tag-action 0",
             "run-.-tag-action 1",
             "run-.-tag-action 2",
             "run-.-tag-action 3",
             "run-.-tag-action 4",
             "run-.-tag-action 5",
             "run-.-tag-action 6",
             "run-.-tag-action 7",
             "run-.-tag-action 8",
             "run-.-tag-action 9",
             ]
models = ["base-1", "base-2", "base-3", "base-4",
          "base-5", "base-6", "base-7", "base-8"]

models = ["even_4", "even_5", "even_6", "even_8"]

for model in models:
    for i in filenames:
        with open(f"Action-Use-Data/{model}/{i}.csv") as original_file:
            csv_reader = csv.reader(original_file, delimiter=',')
            clean_action_choice_data(csv_reader, i, model)

# Produce box plots for action choice frequency

long_models = ["base-6", "base-7", "base-8"]
# or 50
create_paired_boxplots([200, 600], models, ["2 Million", "5 Million"])
# create_boxplots(200, models, "2 Million")
# create_boxplots(600, models, "5 Million")

# for i in range(0, 2000, 50):    # create_boxplots(i+500, models, i+1000)
#
#     create_boxplots(i, models, i)

# Produce running average action frequency graphs
# create_action_plots(0)
# create_action_plots(1)
# create_action_plots(6)

# create_action_plots_unrestricted(0, long_models)
# create_action_plots_unrestricted(1, long_models)
# create_action_plots_unrestricted(6, long_models)




