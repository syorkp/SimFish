import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_metrics(prey_capture, pred_avoidance, window):
    prey_steps = [float(d[1]) for d in prey_capture[1:]]
    pred_steps = [float(d[1]) for d in pred_avoidance[1:]]
    prey_index = [float(d[2]) for d in prey_capture[1:]]
    pred_index = [float(d[2]) for d in pred_avoidance[1:]]

    prey_index_ra = [np.mean(prey_index[i:i+window]) for i, d in enumerate(prey_index) if i <len(prey_index)-window]
    prey_steps = pred_steps[:-window]

    # sns.lineplot(x=prey_steps, y=prey_index_ra)
    sns.set()
    plt.figure(figsize=(8,6))
    plt.plot(prey_steps, prey_index_ra)
    plt.hlines(0.2, 0, 10000,  linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Prey Caught (proportion)", fontsize=20)
    plt.show()

    pred_index_ra = [np.mean(pred_index[i:i+window]) for i, d in enumerate(pred_index) if i <len(pred_index)-window]
    pred_steps = pred_steps[:-window]

    plt.figure(figsize=(8,6))
    plt.plot(pred_steps, pred_index_ra)
    plt.hlines(100, 0, 10000, linestyles={'dashed'}, colors=["y"])
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Predator Avoidance Index", fontsize=20)
    plt.show()

with open('run-.-tag-prey capture index (fraction caught).csv', newline='') as f:
    reader = csv.reader(f)
    prey = list(reader)


with open('run-.-tag-predator avoidance index (avoided_p_pred).csv', newline='') as f:
    reader = csv.reader(f)
    pred = list(reader)

plot_metrics(prey, pred, 20)