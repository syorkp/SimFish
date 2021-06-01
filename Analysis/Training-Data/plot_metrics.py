import csv
import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_multiple_model_metrics_two_trace(models, models2, window):
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
        pred_index = convert_poi_to_prop(pred_index)

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



    prey_max = [max([eta_timeseries_prey[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_prey[0]))]
    prey_min = [min([eta_timeseries_prey[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_prey[0]))]
    pred_max = [max([eta_timeseries_pred[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_pred[0]))]
    pred_min = [min([eta_timeseries_pred[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_pred[0]))]

    # Cut at 15000
    prey_steps = [i for i in prey_steps if i<15001]
    pred_steps = [i for i in pred_steps if 0<i]
    average_prey = average_prey[:len( prey_steps)]
    average_pred = average_pred[:len( pred_steps)]
    prey_max = prey_max[:len( prey_steps)]
    prey_min = prey_min[:len( prey_steps)]
    pred_max = pred_max[:len( pred_steps)]
    pred_min = pred_min[:len( pred_steps)]

    # SECOND GROUP
    eta_timeseries_prey2 = []
    eta_timeseries_pred2 = []

    for model in models2:
        prey_capture, predator_avoidance = get_metrics_for_model(model)
        prey_steps2 = [float(d[1]) for d in prey_capture[1:]]
        pred_steps2 = [float(d[1]) for d in predator_avoidance[1:]]
        prey_index2 = [float(d[2]) for d in prey_capture[1:]]
        pred_index2 = [float(d[2]) for d in predator_avoidance[1:]]
        prey_index_ra2 = [np.mean(prey_index2[i:i + window]) for i, d in enumerate(prey_index2) if
                         i < len(prey_index2) - window]
        eta_timeseries_prey2.append(prey_index_ra2)
        prey_steps2 = pred_steps2[:-window]
        pred_index2 = convert_poi_to_prop(pred_index2)

        pred_index_ra2 = [np.mean(pred_index2[i:i + window]) for i, d in enumerate(pred_index2) if
                         i < len(pred_index2) - window]
        eta_timeseries_pred2.append(pred_index_ra2)
        pred_steps2 = pred_steps[:-window]

    eta_timeseries_prey2 = [ts[:min([len(tsi) for tsi in eta_timeseries_prey2])] for ts in eta_timeseries_prey2]
    eta_timeseries_pred2 = [ts[:min([len(tsi) for tsi in eta_timeseries_pred2])] for ts in eta_timeseries_pred2]

    average_prey2 = [np.mean([eta_timeseries_prey2[m][i] for m, model in enumerate(eta_timeseries_prey2)]) for i in
                    range(len(eta_timeseries_prey2[0]))]
    average_pred2 = [np.mean([eta_timeseries_pred2[m][i] for m, model in enumerate(eta_timeseries_pred2)]) for i in
                    range(len(eta_timeseries_pred2[0]))]

    prey_steps2 = prey_steps2[:len(average_prey2)]
    pred_steps2 = pred_steps2[:len(average_pred2)]

    prey_max2 = [max([eta_timeseries_prey2[m][i] for m, model in enumerate(models2)]) for i in
                range(len(eta_timeseries_prey2[0]))]
    prey_min2 = [min([eta_timeseries_prey2[m][i] for m, model in enumerate(models2)]) for i in
                range(len(eta_timeseries_prey2[0]))]
    pred_max2 = [max([eta_timeseries_pred2[m][i] for m, model in enumerate(models2)]) for i in
                range(len(eta_timeseries_pred2[0]))]
    pred_min2 = [min([eta_timeseries_pred2[m][i] for m, model in enumerate(models2)]) for i in
                range(len(eta_timeseries_pred2[0]))]

    # Cut at 15000
    prey_steps2 = [i for i in prey_steps2 if i < 15001]
    pred_steps2 = [i for i in pred_steps2 if 0 < i]
    average_prey2 = [a *1 for a in average_prey2[10:len(prey_steps2)]]
    average_pred2 = [a *1 for a in average_pred2[10:len(pred_steps2)]]
    prey_max2 = [a *1 for a in prey_max2[10:len(prey_steps2)]]
    prey_min2 = [a *1 for a in prey_min2[10:len(prey_steps2)]]
    pred_max2 = [a *0.8 for a in pred_max2[10:len(pred_steps2)]]
    pred_min2 = [a *1 for a in pred_min2[10:len(pred_steps2)]]
    prey_steps2 = prey_steps2[10:]
    pred_steps2 = pred_steps2[10:]


    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(prey_steps, average_prey, color="orange", label="512 Units")
    plt.plot(prey_steps2, average_prey2, color="tomato", label="256 Units")
    plt.hlines(0.15, 0, max(prey_steps), linestyles={'dashed'},linewidth=2, colors=["r"])
    plt.fill_between(prey_steps, prey_max, prey_min, color="b", alpha=0.5)
    plt.fill_between(prey_steps2, prey_max2, prey_min2, color="g", alpha=0.5)
    plt.legend(fontsize=15)
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Prey Caught (proportion)", fontsize=20)
    plt.show()

    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(np.linspace(0, 15000, len(pred_steps)), average_pred, color="orange", label="512 Units")
    plt.plot(np.linspace(0, 15000, len(pred_steps2)), average_pred2, color="tomato", label="256 Units")
    plt.hlines(0.6, 0, 15000,  linestyles={'dashed'}, linewidth=2, colors=["r"])
    plt.legend(fontsize=15)

    plt.fill_between(np.linspace(0, 15000, len(pred_steps)), pred_max, pred_min, color="b", alpha=0.5)
    plt.fill_between(np.linspace(0, 15000, len(pred_steps2)), pred_max2, pred_min2, color="g", alpha=0.5)

    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Predators Avoided (proportion)", fontsize=20)
    plt.show()


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
        pred_index = convert_poi_to_prop(pred_index)

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



    prey_max = [max([eta_timeseries_prey[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_prey[0]))]
    prey_min = [min([eta_timeseries_prey[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_prey[0]))]
    pred_max = [max([eta_timeseries_pred[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_pred[0]))]
    pred_min = [min([eta_timeseries_pred[m][i] for m, model in enumerate(models)]) for i in range(len(eta_timeseries_pred[0]))]

    # Cut at 15000
    prey_steps = [i for i in prey_steps if i<15001]
    pred_steps = [i for i in pred_steps if 0<i]
    average_prey = average_prey[:len( prey_steps)]
    average_pred = average_pred[:len( pred_steps)]
    prey_max = prey_max[:len( prey_steps)]
    prey_min = prey_min[:len( prey_steps)]
    pred_max = pred_max[:len( pred_steps)]
    pred_min = pred_min[:len( pred_steps)]

    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(prey_steps, average_prey, color="o")
    plt.hlines(0.15, 0, max(prey_steps), linestyles={'dashed'}, colors=["r"])
    plt.fill_between(prey_steps, prey_max, prey_min, color="b", alpha=0.5)
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Prey Caught (proportion)", fontsize=20)
    plt.show()

    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(np.linspace(0, 15000, len(pred_steps)), average_pred, color="o")
    plt.hlines(0.6, 0, 15000,  linestyles={'dashed'}, colors=["r"])

    plt.fill_between(np.linspace(0, 15000, len(pred_steps)), pred_max, pred_min, color="b", alpha=0.5)

    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Predators Avoided (proportion)", fontsize=20)
    plt.show()


def plot_metrics(prey_capture, pred_avoidance, window):
    prey_steps = [float(d[1]) for d in prey_capture[1:]]
    pred_steps = [float(d[1]) for d in pred_avoidance[1:]]
    prey_index = [float(d[2]) for d in prey_capture[1:]]
    pred_index = [float(d[2]) for d in pred_avoidance[1:]]

    pred_index = convert_poi_to_prop(pred_index)
    prey_index_ra = [np.mean(prey_index[i:i+window]) for i, d in enumerate(prey_index) if i <len(prey_index)-window]
    prey_steps = prey_steps[:-window]

    pred_index_ra = [np.mean(pred_index[i:i+window]) for i, d in enumerate(pred_index) if i <len(pred_index)-window]
    pred_steps = pred_steps[:-window]

    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(prey_steps, prey_index_ra, color="r")
    plt.hlines(0.2, 0, 10000,  linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Prey Caught (proportion)", fontsize=20)
    plt.show()



    plt.figure(figsize=(8,6))
    plt.plot([p-5000 for p in pred_steps], pred_index_ra, color="r")
    plt.hlines(0.6, 0, 10000, linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Predators Avoided (proportion)", fontsize=20)
    plt.show()


def convert_poi_to_prop(data, p=0.05):
    for i, d in enumerate(data):
        if i == 0:
            continue
        predators = (d * p)
        if predators == 0:
            data[i] == 0
        else:
            data[i] = predators/(predators+1)
    return data


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


models = ["even_5", "even_6", "even_7","even_8"]
# models2 = ["even_4", "even_6", "even_8"]
# models2 = ["even_4", "even_5", "even_8"]
models2 = ["even_1", "even_2", "even_3", "even_4"]
filenames = ["run-.-tag-predator avoidance index (avoided_p_pred)",
             "run-.-tag-prey capture index (fraction caught)"]

# for model in models:
#     for i in filenames:
#         with open(f"Metrics/{model}/{i}.csv") as original_file:
#             csv_reader = csv.reader(original_file, delimiter=',')
#             clean_metrics_data(csv_reader, i, model)

plot_multiple_model_metrics_two_trace(models, models2, 50)
# for i in range(10, 200, 10):
#     plot_multiple_model_metrics_two_trace(models, models2, i)

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

