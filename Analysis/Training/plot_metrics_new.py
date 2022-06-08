import numpy as np
import matplotlib.pyplot as plt
import csv

from Analysis.Training.plot_metrics import get_metrics_for_model


def get_config_change_points(model_name):
    with open(f'../Data/Metrics/{model_name}/run-.-tag-Configuration change.csv', newline='') as f:
        reader = csv.reader(f)
        config_change_points = list(reader)
    return config_change_points


def plot_pci_single(model_name, window):
    PCI, PAI = get_metrics_for_model(model_name)
    total_steps_recorded = float(PCI[-1][1])
    config_switching_points = get_config_change_points(model_name)
    steps_change_occurs = [float(d[1]) for d in config_switching_points[1:]]
    normalised_switching_points = [s/total_steps_recorded for s in steps_change_occurs]

    prey_capture_index = [float(d[2]) for d in PCI[1:]]
    prey_capture_index_rolling_average = [np.mean(prey_capture_index[i:i + window]) for i, d in enumerate(prey_capture_index) if i < len(prey_capture_index) - window]
    max_pci = max(prey_capture_index_rolling_average)

    # Plotting the data
    plt.figure(figsize=(15, 10))
    plt.plot(np.linspace(0, 1, len(prey_capture_index_rolling_average)), prey_capture_index_rolling_average)
    plt.title("PCI Rolling Average")
    plt.xlabel("Point in Training (Normalised)")
    plt.ylabel("PCI")
    for c in normalised_switching_points:
        # TODO: Add contingency to only plot important switching points.
        plt.vlines(c, 0, max_pci, colors="r")
    plt.show()

plot_pci_single("dqn_scaffold_14-1", 10)