import csv
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




def plot_multiple_model_metrics(models, window):
    eta_timeseries_prey = []
    eta_timeseries_pred = []

    for model in models:
        prey_capture, predator_avoidance = get_metrics_for_model(model)
        prey_steps = [float(d[1]) for d in prey_capture[1:]]
        pred_steps = [float(d[1]) for d in predator_avoidance[1:]]
        prey_index = [float(d[2]) for d in prey_capture[1:]]
        pred_index = [float(d[2]) for d in predator_avoidance[1:]]
        prey_index_ra = [np.mean(prey_index[i:i+window]) for i, d in enumerate(prey_index) if i <len(prey_index)-window]
        eta_timeseries_prey.append(prey_index_ra)
        prey_steps = pred_steps[:-window]
        pred_index_ra = [np.mean(pred_index[i:i + window]) for i, d in enumerate(pred_index) if
                         i < len(pred_index) - window]
        eta_timeseries_pred.append(pred_index_ra)
        pred_steps = pred_steps[:-window]

    eta_timeseries_prey = [ts[:min([len(tsi) for tsi in eta_timeseries_prey])] for ts in eta_timeseries_prey]
    eta_timeseries_pred = [ts[:min([len(tsi) for tsi in eta_timeseries_pred])] for ts in eta_timeseries_pred]

    average_prey = [np.mean([eta_timeseries_prey[m][i] for m, model in enumerate(eta_timeseries_prey)]) for i in range(len(eta_timeseries_prey[0]))]
    average_pred = [np.mean([eta_timeseries_pred[m][i] for m, model in enumerate(eta_timeseries_pred)]) for i in range(len(eta_timeseries_pred[0]))]

    prey_steps = prey_steps[:len(average_prey)]
    pred_steps = pred_steps[:len(average_pred)]


    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(prey_steps, average_prey)
    plt.hlines(0.2, 0, 10000,  linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Prey Caught (proportion)", fontsize=20)
    plt.show()

    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(pred_steps, average_pred)
    plt.hlines(0.2, 0, 10000,  linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Prey Caught (proportion)", fontsize=20)
    plt.show()


def plot_metrics(prey_capture, pred_avoidance, window, prey_ebars, pred_ebars):
    prey_steps = [float(d[1]) for d in prey_capture[1:]]
    pred_steps = [float(d[1]) for d in pred_avoidance[1:]]
    prey_index = [float(d[2]) for d in prey_capture[1:]]
    pred_index = [float(d[2]) for d in pred_avoidance[1:]]

    prey_index_ra = [np.mean(prey_index[i:i+window]) for i, d in enumerate(prey_index) if i <len(prey_index)-window]
    prey_steps = prey_steps[:-window]

    pred_index_ra = [np.mean(pred_index[i:i+window]) for i, d in enumerate(pred_index) if i <len(pred_index)-window]
    pred_steps = pred_steps[:-window]

    # Rnadom example
    # prey_ebars = np.random.uniform(0.07, 0.15, [len(prey_steps), 2])
    # pred_ebars = np.random.uniform(40, 75, [len(pred_steps), 2])
    # prey_ebars = np.array([prey_ebars[i//random.randint(10, 30)] for i in range(len(prey_ebars))])
    # pred_ebars = np.array([pred_ebars[i//random.randint(10, 30)] for i in range(len(pred_ebars))])

    # sns.lineplot(x=prey_steps, y=prey_index_ra)
    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(prey_steps, prey_index_ra, color="r")
    plt.fill_between(prey_steps, [prey_index_ra[i]-prey_ebars[i, 0] if prey_index_ra[i]-prey_ebars[i, 0]>0 else 0 for i in range(len(prey_ebars))],
                     [prey_index_ra[i]+prey_ebars[i, 1] for i in range(len(prey_ebars))], color="b")
    plt.hlines(0.2, 0, 10000,  linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Prey Caught (proportion)", fontsize=20)
    plt.show()



    plt.figure(figsize=(8,6))
    plt.plot(pred_steps, pred_index_ra, color="r")
    plt.fill_between(pred_steps, [pred_index_ra[i]-pred_ebars[i, 0] if pred_index_ra[i]-pred_ebars[i, 0]>0 else 0  for i in range(len(pred_ebars))],
                     [pred_index_ra[i]+pred_ebars[i, 1] for i in range(len(pred_ebars))], color="b")
    plt.hlines(100, 0, 10000, linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Predator Avoidance Index", fontsize=20)
    plt.show()


def get_metrics_for_model(model_name):
    with open(f'Metrics/{model_name}/run-.-tag-prey capture index (fraction caught).csv', newline='') as f:
        reader = csv.reader(f)
        prey = list(reader)

    with open(f'Metrics/{model_name}/run-.-tag-predator avoidance index (avoided_p_pred).csv', newline='') as f:
        reader = csv.reader(f)
        pred = list(reader)
    return prey, pred


def clean_metrics_data(reader, file_name, model):
    increment = 0
    previous_step = 0
    with open(f"Metrics/{model}/{file_name}.csv", "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        # writer.writerow(["Wall time", "Step", "Value"])
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


models = ["even_5", "even_6", "even_8"]
filenames = ["run-.-tag-predator avoidance index (avoided_p_pred)",
             "run-.-tag-prey capture index (fraction caught)"]

for model in models:
    for i in filenames:
        with open(f"Metrics/{model}/{i}.csv") as original_file:
            csv_reader = csv.reader(original_file, delimiter=',')
            clean_metrics_data(csv_reader, i, model)


# plot_multiple_model_metrics(models, 20)
#
# prey, pred = get_metrics_for_model("even_4")
# plot_metrics(prey, pred, 100)
# #
#
# prey, pred = get_metrics_for_model("even_5")
# plot_metrics(prey, pred, 20)
#
# prey, pred = get_metrics_for_model("even_6")
# plot_metrics(prey, pred, 20)
# prey, pred = get_metrics_for_model("even_8")
# plot_metrics(prey, pred, 20)

