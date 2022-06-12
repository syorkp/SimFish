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


def plot_pci_multiple_models(model_names, window, window2=4):
    """
    Uses the first data point as a line. for all others, uses to create error bars.
    :return:
    """
    # Trace from all models.
    PCI_dataset_for_each = [get_metrics_for_model(model)[0] for model in model_names]
    steps_for_each = [[int(model_data[i][1]) for i, m in enumerate(model_data) if i != 0] for model_data in PCI_dataset_for_each]
    PCI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in PCI_dataset_for_each]

    # Get num steps to use.
    steps_to_use = min([max(s) for s in steps_for_each])
    PCI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data in enumerate(PCI_for_each)]
    PCI_for_each = [[np.mean(PCI_for_each[m][i:i + window]) for i, d in
                                          enumerate(PCI_for_each[m]) if i < len(PCI_for_each[m]) - window] for m in range(len(model_names))]




    steps_for_each = [[step for step in steps if step <= steps_to_use] for steps in steps_for_each]
    steps_for_each = [s[:-window] for s in steps_for_each]

    for m in range(len(model_names)):
        plt.plot(steps_for_each[m], PCI_for_each[m])
    plt.show()

    flattened_steps = sorted([x for xi in steps_for_each for x in xi])

    max_pci_all_steps = []
    min_pci_all_steps = []
    min_line = []
    max_line = []
    for i, s in enumerate(flattened_steps):
        PCIs = []
        lines = []
        for m in range(len(model_names)):
            if s in steps_for_each[m]:
                pci_at_step = PCI_for_each[m][steps_for_each[m].index(s)]
                PCIs.append(pci_at_step)
                lines.append(m)
            else:
                pass

        max_pci_all_steps.append(max(PCIs))
        max_line.append(lines[PCIs.index(max(PCIs))])
        min_pci_all_steps.append(min(PCIs))
        min_line.append(lines[PCIs.index(min(PCIs))])

    # To clean up error bars, go through each trace, following that one along while it is the minimum.
    new_min_line = []
    new_max_line = []
    for i, s in enumerate(flattened_steps):
        if i < 10:
            pass
        else:
            min_in_window = min(min_pci_all_steps[i-10:i])
            new_min_line.append(min_in_window)
            max_in_window = max(max_pci_all_steps[i-10:i])
            new_max_line.append(max_in_window)

    # Main trace
    model_name = model_names[0]
    PCI, PAI = get_metrics_for_model(model_name)
    total_steps_recorded = float(PCI[-1][1])
    config_switching_points = get_config_change_points(model_name)
    steps_change_occurs = [float(d[1]) for d in config_switching_points[1:]]
    normalised_switching_points = [s/total_steps_recorded for s in steps_change_occurs]

    prey_capture_index = [float(d[2]) for d in PCI[1:]]
    prey_capture_index_rolling_average = [np.mean(prey_capture_index[i:i + window]) for i, d in
                                          enumerate(prey_capture_index) if i < len(prey_capture_index) - window]
    max_pci = max(prey_capture_index_rolling_average)

    # Plotting the data
    plt.figure(figsize=(15, 10))
    plt.plot(steps_for_each[1], PCI_for_each[1], color="r")
    plt.title("PCI Rolling Average")
    plt.xlabel("Point in Training (Normalised)")
    plt.ylabel("PCI")
    for c in normalised_switching_points:
        # TODO: Add contingency to only plot important switching points.
        plt.vlines(c, 0, max_pci, colors="r")




    plt.fill_between(flattened_steps[:-window2], min_pci_all_steps, max_pci_all_steps, color="b", alpha=0.5)

    plt.show()


# plot_pci_single("dqn_scaffold_20-1", 30)
# plot_pci_single("dqn_scaffold_20-2", 30)
# plot_pci_single("dqn_scaffold_21-1", 30)
# plot_pci_single("dqn_scaffold_21-2", 30)
plot_pci_multiple_models(["dqn_scaffold_20-1", "dqn_scaffold_20-2", "dqn_scaffold_21-1", "dqn_scaffold_21-2"], 30)
# plot_pci_single("dqn_scaffold_20-1", 30)