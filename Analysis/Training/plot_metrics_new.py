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


def plot_pci_multiple_models(model_names, window, scaffold_points, scaffold_point_names, window2=20):
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
        if i < window2:
            pass
        else:
            min_in_window = min(min_pci_all_steps[i-window2:i])
            new_min_line.append(min_in_window)
            max_in_window = max(max_pci_all_steps[i-window2:i])
            new_max_line.append(max_in_window)
    flattened_steps = flattened_steps[window2:]

    # Main trace
    model_name = model_names[0]
    PCI, PAI = get_metrics_for_model(model_name)
    total_steps_recorded = float(PCI[-1][1])
    config_switching_points = get_config_change_points(model_name)
    steps_change_occurs = [float(d[1]) for d in config_switching_points[1:]]
    steps_change_occurs = [steps_change_occurs[i] for i in scaffold_points]
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
    for i, c in enumerate(steps_change_occurs):
        plt.vlines(c, 0, max_pci, colors="r")
        plt.text(c, 0.01*i, scaffold_point_names[i])

    plt.fill_between(flattened_steps, new_min_line, new_max_line, color="b", alpha=0.5)

    plt.show()


def plot_pai_multiple_models(model_names, window, scaffold_points, scaffold_point_names, window2=20):
    """
    Uses the first data point as a line. for all others, uses to create error bars.
    :return:
    """
    # Trace from all models.
    PAI_dataset_for_each = [get_metrics_for_model(model)[1] for model in model_names]
    steps_for_each = [[int(model_data[i][1]) for i, m in enumerate(model_data) if i != 0] for model_data in PAI_dataset_for_each]
    PAI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in PAI_dataset_for_each]

    # Get num steps to use.
    steps_to_use = min([max(s) for s in steps_for_each])
    PAI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data in enumerate(PAI_for_each)]
    PAI_for_each = [[np.mean(PAI_for_each[m][i:i + window]) for i, d in
                                          enumerate(PAI_for_each[m]) if i < len(PAI_for_each[m]) - window] for m in range(len(model_names))]

    steps_for_each = [[step for step in steps if step <= steps_to_use] for steps in steps_for_each]
    steps_for_each = [s[:-window] for s in steps_for_each]

    flattened_steps = sorted([x for xi in steps_for_each for x in xi])

    max_pai_all_steps = []
    min_pai_all_steps = []
    min_line = []
    max_line = []
    for i, s in enumerate(flattened_steps):
        PCIs = []
        lines = []
        for m in range(len(model_names)):
            if s in steps_for_each[m]:
                pci_at_step = PAI_for_each[m][steps_for_each[m].index(s)]
                PCIs.append(pci_at_step)
                lines.append(m)
            else:
                pass

        max_pai_all_steps.append(max(PCIs))
        max_line.append(lines[PCIs.index(max(PCIs))])
        min_pai_all_steps.append(min(PCIs))
        min_line.append(lines[PCIs.index(min(PCIs))])

    # To clean up error bars, go through each trace, following that one along while it is the minimum.
    new_min_line = []
    new_max_line = []
    for i, s in enumerate(flattened_steps):
        if i < window2:
            pass
        else:
            min_in_window = min(min_pai_all_steps[i-window2:i])
            new_min_line.append(min_in_window)
            max_in_window = max(max_pai_all_steps[i-window2:i])
            new_max_line.append(max_in_window)
    flattened_steps = flattened_steps[window2:]

    # Main trace
    model_name = model_names[0]
    PCI, PAI = get_metrics_for_model(model_name)
    total_steps_recorded = float(PAI[-1][1])
    config_switching_points = get_config_change_points(model_name)
    steps_change_occurs = [float(d[1]) for d in config_switching_points[1:]]
    steps_change_occurs = [steps_change_occurs[i] for i in scaffold_points]
    normalised_switching_points = [s/total_steps_recorded for s in steps_change_occurs]

    predator_avoidance_index = [float(d[2]) for d in PAI[1:]]
    predator_avoidance_index_rolling_average = [np.mean(predator_avoidance_index[i:i + window]) for i, d in
                                          enumerate(predator_avoidance_index) if i < len(predator_avoidance_index) - window]
    max_pci = max(predator_avoidance_index_rolling_average)

    # Plotting the data
    plt.figure(figsize=(15, 10))
    plt.plot(steps_for_each[1], PAI_for_each[1], color="r")
    plt.title("PAI Rolling Average")
    plt.xlabel("Episode")
    plt.ylabel("PCI")
    for i, c in enumerate(steps_change_occurs):
        plt.vlines(c, 0, max_pci, colors="r")
        plt.text(c, 0.01*i, scaffold_point_names[i])

    plt.fill_between(flattened_steps, new_min_line, new_max_line, color="b", alpha=0.5)

    plt.show()


def plot_metric_multiple_models(metric_list, model_names, window, scaffold_points, scaffold_point_names, window2=20,
                                metric_name=None, failed_list=None):
    steps_for_each = [[int(model_data[i][1]) for i, m in enumerate(model_data) if i != 0] for model_data in
                      metric_list]
    PAI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    metric_list]

    # Get num steps to use.
    steps_to_use = min([max(s) for s in steps_for_each])
    PAI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PAI_for_each)]
    PAI_for_each = [[np.mean(PAI_for_each[m][i:i + window]) for i, d in
                     enumerate(PAI_for_each[m]) if i < len(PAI_for_each[m]) - window] for m in range(len(model_names))]

    steps_for_each = [[step for step in steps if step <= steps_to_use] for steps in steps_for_each]
    steps_for_each = [s[:-window] for s in steps_for_each]

    flattened_steps = sorted([x for xi in steps_for_each for x in xi])

    max_pai_all_steps = []
    min_pai_all_steps = []
    min_line = []
    max_line = []
    for i, s in enumerate(flattened_steps):
        PCIs = []
        lines = []
        for m in range(len(metric_list)):
            if s in steps_for_each[m]:
                pci_at_step = PAI_for_each[m][steps_for_each[m].index(s)]
                PCIs.append(pci_at_step)
                lines.append(m)
            else:
                pass

        max_pai_all_steps.append(max(PCIs))
        max_line.append(lines[PCIs.index(max(PCIs))])
        min_pai_all_steps.append(min(PCIs))
        min_line.append(lines[PCIs.index(min(PCIs))])

    # To clean up error bars, go through each trace, following that one along while it is the minimum.
    new_min_line = []
    new_max_line = []
    for i, s in enumerate(flattened_steps):
        if i < window2:
            pass
        else:
            min_in_window = min(min_pai_all_steps[i - window2:i])
            new_min_line.append(min_in_window)
            max_in_window = max(max_pai_all_steps[i - window2:i])
            new_max_line.append(max_in_window)
    flattened_steps = flattened_steps[window2:]

    # Main trace
    model_name = model_names[0]
    # PCI, PAI, _, __ = get_metrics_for_model(model_name)
    PAI = PAI_for_each[0]
    config_switching_points = get_config_change_points(model_name)
    steps_change_occurs = [float(d[1]) for d in config_switching_points[1:]]
    steps_change_occurs = [steps_change_occurs[i] for i in scaffold_points]

    predator_avoidance_index = PAI
    predator_avoidance_index_rolling_average = [np.mean(predator_avoidance_index[i:i + window]) for i, d in
                                                enumerate(predator_avoidance_index) if
                                                i < len(predator_avoidance_index) - window]
    max_pci = max(predator_avoidance_index_rolling_average)

    # Plotting the data
    plt.figure(figsize=(15, 10))
    plt.plot(steps_for_each[0], PAI_for_each[0], color="r")
    plt.title(f"{metric_name} Rolling Average", fontsize=25)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel(metric_name, fontsize=25)
    for i, c in enumerate(steps_change_occurs):
        plt.vlines(c, 0, max_pci, colors="orange", linestyles="dashed")
        plt.text(c, max_pci/20 * i, scaffold_point_names[i], fontsize=18)

    plt.fill_between(flattened_steps, new_min_line, new_max_line, color="b", alpha=0.5)


    # Plot additional trials
    if failed_list is not None:
        for i in range(len(failed_list)):
            steps_for_each = [[int(model_data[i][1]) for i, m in enumerate(model_data) if i != 0] for model_data in
                              metric_list]
            PAI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                            metric_list]
            plt.plot(steps_for_each[i], PAI_for_each[i], color="y")

    plt.savefig(f"../Figures/Panels/Panel-3/{metric_name}")
    plt.show()


def plot_all_metrics_multiple_models(model_list, scaffold_points, scaffold_point_names, main_window, failed_list=None):
    if failed_list is not None:
        PCI_dataset_for_each_failed = [get_metrics_for_model(model)[0] for model in failed_list]
        PAI_dataset_for_each_failed = [get_metrics_for_model(model)[1] for model in failed_list]
        CSR_dataset_for_each_failed = [get_metrics_for_model(model)[2] for model in failed_list]
        PCR_dataset_for_each_failed = [get_metrics_for_model(model)[3] for model in failed_list]
    else:
        PCI_dataset_for_each_failed = None
        PAI_dataset_for_each_failed = None
        CSR_dataset_for_each_failed = None
        PCR_dataset_for_each_failed = None


    PCI_dataset_for_each = [get_metrics_for_model(model)[0] for model in model_list]
    plot_metric_multiple_models(PCI_dataset_for_each, model_list, main_window, scaffold_points, scaffold_point_names,
                                metric_name="PCI", failed_list=PCI_dataset_for_each_failed)

    PAI_dataset_for_each = [get_metrics_for_model(model)[1] for model in model_list]
    plot_metric_multiple_models(PAI_dataset_for_each, model_list, main_window, scaffold_points, scaffold_point_names,
                                metric_name="PAI", failed_list=PAI_dataset_for_each_failed)

    CSR_dataset_for_each = [get_metrics_for_model(model)[2] for model in model_list]
    plot_metric_multiple_models(CSR_dataset_for_each, model_list, main_window, scaffold_points, scaffold_point_names,
                                metric_name="CSR", failed_list=CSR_dataset_for_each_failed)

    PCR_dataset_for_each = [get_metrics_for_model(model)[3] for model in model_list]
    plot_metric_multiple_models(PCR_dataset_for_each, model_list, main_window, scaffold_points, scaffold_point_names,
                                metric_name="PCR", failed_list=PCR_dataset_for_each_failed)


def rescale_by_config_change_points(metrics, config_change_points, original_episodes):
    full_length = len(metrics[0])
    length_per_switch = full_length/len(config_change_points)

    count = 0
    previous_index = 0
    new_steps = np.array([])
    for switch in config_change_points:
        num_points = np.findoriginal_episodes.index(int(switch)) - previous_index
        intermediate_points = np.linspace(count, count + length_per_switch, num_points, endpoint=False)
        count += length_per_switch
        previous_index += original_episodes.index(int(switch))
        new_steps += intermediate_points
    x = True


def plot_four_metrics(models, scaffold_points, scaffold_point_names, window):
    PCI_dataset_for_each = [get_metrics_for_model(model)[0] for model in model_list]
    PAI_dataset_for_each = [get_metrics_for_model(model)[1] for model in model_list]
    CSR_dataset_for_each = [get_metrics_for_model(model)[2] for model in model_list]
    PCR_dataset_for_each = [get_metrics_for_model(model)[3] for model in model_list]

    steps_for_each = [[int(model_data[i][1]) for i, m in enumerate(model_data) if i != 0] for model_data in
                      PCI_dataset_for_each]
    steps_to_use = min([max(s) for s in steps_for_each])

    PCI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PCI_dataset_for_each]
    PAI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PAI_dataset_for_each]
    CSR_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    CSR_dataset_for_each]
    PCR_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PCR_dataset_for_each]

    PCI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PCI_for_each)]
    PCI_for_each = [[np.mean(PCI_for_each[m][i:i + window]) for i, d in
                     enumerate(PCI_for_each[m]) if i < len(PCI_for_each[m]) - window] for m in range(len(models))]
    PAI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PAI_for_each)]
    PAI_for_each = [[np.mean(PAI_for_each[m][i:i + window]) for i, d in
                     enumerate(PAI_for_each[m]) if i < len(PAI_for_each[m]) - window] for m in range(len(models))]
    CSR_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(CSR_for_each)]
    CSR_for_each = [[np.mean(CSR_for_each[m][i:i + window]) for i, d in
                     enumerate(CSR_for_each[m]) if i < len(CSR_for_each[m]) - window] for m in range(len(models))]
    PCR_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PCR_for_each)]
    PCR_for_each = [[np.mean(PCR_for_each[m][i:i + window]) for i, d in
                     enumerate(PCR_for_each[m]) if i < len(PCR_for_each[m]) - window] for m in range(len(models))]

    steps_for_each = [[step for step in steps if step <= steps_to_use] for steps in steps_for_each]
    steps_for_each = [s[:-window] for s in steps_for_each]

    fig, axs = plt.subplots(4, 1, figsize=(8, 16), sharex=True)
    for model in range(len(models)):
        if model == 0:
            color = "r"
        else:
            color = "black"

        axs[0].plot(metrics, metrics[0], c=color)
        axs[0].set_ylabel("PCI")
        axs[1].plot(steps_for_each[model], metrics[1], c=color)
        axs[2].plot(steps_for_each[model], metrics[2], c=color)
        axs[3].plot(steps_for_each[model], metrics[3], c=color)
    plt.show()


def plot_four_metrics_error_bars(models, scaffold_points, scaffold_point_names, window, window2=20):
    PCI_dataset_for_each = [get_metrics_for_model(model)[0] for model in model_list]
    PAI_dataset_for_each = [get_metrics_for_model(model)[1] for model in model_list]
    CSR_dataset_for_each = [get_metrics_for_model(model)[2] for model in model_list]
    PCR_dataset_for_each = [get_metrics_for_model(model)[3] for model in model_list]

    steps_for_each = [[int(model_data[i][1]) for i, m in enumerate(model_data) if i != 0] for model_data in
                      PCI_dataset_for_each]
    steps_to_use = min([max(s) for s in steps_for_each])

    PCI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PCI_dataset_for_each]
    PAI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PAI_dataset_for_each]
    CSR_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    CSR_dataset_for_each]
    PCR_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PCR_dataset_for_each]

    PCI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PCI_for_each)]
    PCI_for_each = [[np.mean(PCI_for_each[m][i:i + window]) for i, d in
                     enumerate(PCI_for_each[m]) if i < len(PCI_for_each[m]) - window] for m in range(len(models))]
    PAI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PAI_for_each)]
    PAI_for_each = [[np.mean(PAI_for_each[m][i:i + window]) for i, d in
                     enumerate(PAI_for_each[m]) if i < len(PAI_for_each[m]) - window] for m in range(len(models))]
    CSR_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(CSR_for_each)]
    CSR_for_each = [[np.mean(CSR_for_each[m][i:i + window]) for i, d in
                     enumerate(CSR_for_each[m]) if i < len(CSR_for_each[m]) - window] for m in range(len(models))]
    PCR_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PCR_for_each)]
    PCR_for_each = [[np.mean(PCR_for_each[m][i:i + window]) for i, d in
                     enumerate(PCR_for_each[m]) if i < len(PCR_for_each[m]) - window] for m in range(len(models))]

    steps_for_each = [[step for step in steps if step <= steps_to_use] for steps in steps_for_each]
    steps_for_each = [s[:-window] for s in steps_for_each]

    flattened_steps = sorted([x for xi in steps_for_each for x in xi])

    new_min_line_all, new_max_line_all = [], []
    flattened_steps_all = []
    for metric_list in [PCI_for_each, PAI_for_each, CSR_for_each, PCR_for_each]:
        max_pai_all_steps = []
        min_pai_all_steps = []
        min_line = []
        max_line = []
        for i, s in enumerate(flattened_steps):
            PCIs = []
            lines = []
            for m in range(len(metric_list)):
                if s in steps_for_each[m]:
                    pci_at_step = metric_list[m][steps_for_each[m].index(s)]
                    PCIs.append(pci_at_step)
                    lines.append(m)
                else:
                    pass

            max_pai_all_steps.append(max(PCIs))
            max_line.append(lines[PCIs.index(max(PCIs))])
            min_pai_all_steps.append(min(PCIs))
            min_line.append(lines[PCIs.index(min(PCIs))])

        new_min_line = []
        new_max_line = []
        for i, s in enumerate(flattened_steps):
            if i < window2:
                pass
            else:
                min_in_window = min(min_pai_all_steps[i - window2:i])
                new_min_line.append(min_in_window)
                max_in_window = max(max_pai_all_steps[i - window2:i])
                new_max_line.append(max_in_window)
        flattened_steps = flattened_steps[window2:]
        flattened_steps_all.append(flattened_steps)
        new_min_line_all.append(new_min_line)
        new_max_line_all.append(new_max_line)

    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
    axs[0].plot(steps_for_each[0], PCI_for_each[0], c="b")
    axs[0].set_ylabel("PCI", fontsize=30)
    axs[0].fill_between(flattened_steps_all[0], new_min_line_all[0], new_max_line_all[0], color="b", alpha=0.3)

    axs[1].plot(steps_for_each[0], PAI_for_each[0], c="b")
    axs[1].set_ylabel("PAI", fontsize=30)
    axs[1].fill_between(flattened_steps_all[1], new_min_line_all[1], new_max_line_all[1], color="b", alpha=0.3)

    axs[2].plot(steps_for_each[0], CSR_for_each[0], c="b")
    axs[2].set_ylabel("CSR", fontsize=30)
    axs[2].fill_between(flattened_steps_all[2], new_min_line_all[2], new_max_line_all[2], color="b", alpha=0.3)

    axs[3].plot(steps_for_each[0], PCR_for_each[0], c="b")
    axs[3].set_ylabel("PRC", fontsize=30)
    axs[3].set_xlabel("Episode", fontsize=30)
    axs[3].fill_between(flattened_steps_all[3], new_min_line_all[3], new_max_line_all[3], color="b", alpha=0.3)
    axs[0].legend(["Example Model", "Range over models"],
                  loc="lower right", fontsize=18)

    for i in range(3):
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)

    axs[3].spines["right"].set_visible(False)
    axs[3].spines["top"].set_visible(False)

    config_switching_points = get_config_change_points(models[0])
    steps_change_occurs = [float(d[1]) for d in config_switching_points[1:]]
    steps_change_occurs_important = [steps_change_occurs[i] for i in scaffold_points]

    # Add all scaffold labels
    for i, c in enumerate(steps_change_occurs):
        if c < steps_to_use-200:
            axs[3].vlines(c, 0, max(new_max_line_all[-1]), colors="orange", linestyles="dashed", linewidth=1)

    # Add main scaffold labels
    for i, c in enumerate(steps_change_occurs_important):
        axs[3].vlines(c, 0, max(new_max_line_all[-1]), colors="orange", linestyles="dashed", linewidth=3)
        axs[3].text(c, max(PCR_for_each[0])/10 * i, scaffold_point_names[i], fontsize=18)

    plt.tight_layout()
    plt.savefig("../Figures/Panels/Panel-3/Full-Metrics.png")
    plt.show()


def plot_four_metrics_multiple_lines(models, scaffold_points, scaffold_point_names, window, window2=20):
    PCI_dataset_for_each = [get_metrics_for_model(model)[0] for model in model_list]
    PAI_dataset_for_each = [get_metrics_for_model(model)[1] for model in model_list]
    CSR_dataset_for_each = [get_metrics_for_model(model)[2] for model in model_list]
    PCR_dataset_for_each = [get_metrics_for_model(model)[3] for model in model_list]

    steps_for_each = [[int(model_data[i][1]) for i, m in enumerate(model_data) if i != 0] for model_data in
                      PCI_dataset_for_each]
    steps_to_use = min([max(s) for s in steps_for_each])

    PCI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PCI_dataset_for_each]
    PAI_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PAI_dataset_for_each]
    CSR_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    CSR_dataset_for_each]
    PCR_for_each = [[float(model_data[i][2]) for i, m in enumerate(model_data) if i != 0] for model_data in
                    PCR_dataset_for_each]

    PCI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PCI_for_each)]
    PCI_for_each = [[np.mean(PCI_for_each[m][i:i + window]) for i, d in
                     enumerate(PCI_for_each[m]) if i < len(PCI_for_each[m]) - window] for m in range(len(models))]
    PAI_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PAI_for_each)]
    PAI_for_each = [[np.mean(PAI_for_each[m][i:i + window]) for i, d in
                     enumerate(PAI_for_each[m]) if i < len(PAI_for_each[m]) - window] for m in range(len(models))]
    CSR_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(CSR_for_each)]
    CSR_for_each = [[np.mean(CSR_for_each[m][i:i + window]) for i, d in
                     enumerate(CSR_for_each[m]) if i < len(CSR_for_each[m]) - window] for m in range(len(models))]
    PCR_for_each = [[pci for i, pci in enumerate(model_data) if steps_for_each[m][i] <= steps_to_use] for m, model_data
                    in enumerate(PCR_for_each)]
    PCR_for_each = [[np.mean(PCR_for_each[m][i:i + window]) for i, d in
                     enumerate(PCR_for_each[m]) if i < len(PCR_for_each[m]) - window] for m in range(len(models))]

    steps_for_each = [[step for step in steps if step <= steps_to_use] for steps in steps_for_each]
    steps_for_each = [s[:-window] for s in steps_for_each]

    flattened_steps = sorted([x for xi in steps_for_each for x in xi])

    new_min_line_all, new_max_line_all = [], []
    flattened_steps_all = []
    for metric_list in [PCI_for_each, PAI_for_each, CSR_for_each, PCR_for_each]:
        max_pai_all_steps = []
        min_pai_all_steps = []
        min_line = []
        max_line = []
        for i, s in enumerate(flattened_steps):
            PCIs = []
            lines = []
            for m in range(len(metric_list)):
                if s in steps_for_each[m]:
                    pci_at_step = metric_list[m][steps_for_each[m].index(s)]
                    PCIs.append(pci_at_step)
                    lines.append(m)
                else:
                    pass

            max_pai_all_steps.append(max(PCIs))
            max_line.append(lines[PCIs.index(max(PCIs))])
            min_pai_all_steps.append(min(PCIs))
            min_line.append(lines[PCIs.index(min(PCIs))])

        new_min_line = []
        new_max_line = []
        for i, s in enumerate(flattened_steps):
            if i < window2:
                pass
            else:
                min_in_window = min(min_pai_all_steps[i - window2:i])
                new_min_line.append(min_in_window)
                max_in_window = max(max_pai_all_steps[i - window2:i])
                new_max_line.append(max_in_window)
        flattened_steps = flattened_steps[window2:]
        flattened_steps_all.append(flattened_steps)
        new_min_line_all.append(new_min_line)
        new_max_line_all.append(new_max_line)

    fig, axs = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    for i in range(len(models)):
        if i == 0:
            c = "b"
        else:
            c = "black"
        axs[0].plot(steps_for_each[i], PCI_for_each[i], c=c)
        axs[1].plot(steps_for_each[i], PAI_for_each[i], c=c)
        axs[2].plot(steps_for_each[i], CSR_for_each[i], c=c)
        axs[3].plot(steps_for_each[i], PCR_for_each[i], c=c)

    axs[0].set_ylabel("PCI", fontsize=30)
    axs[1].set_ylabel("PAI", fontsize=30)
    axs[2].set_ylabel("CSR", fontsize=30)
    axs[3].set_ylabel("PRC", fontsize=30)
    axs[3].set_xlabel("Episode", fontsize=30)
    axs[0].legend(["Example Model", "Other Models"],
                  loc="lower right", fontsize=18)

    axs[0].tick_params(labelsize=14)
    axs[1].tick_params(labelsize=14)
    axs[2].tick_params(labelsize=14)
    axs[3].tick_params(axis="x", labelsize=20)
    axs[3].tick_params(axis="y", labelsize=14)

    for i in range(3):
        axs[i].spines["right"].set_visible(False)
        axs[i].spines["top"].set_visible(False)

    axs[3].spines["right"].set_visible(False)
    axs[3].spines["top"].set_visible(False)

    config_switching_points = get_config_change_points(models[0])
    steps_change_occurs = [float(d[1]) for d in config_switching_points[1:]]
    steps_change_occurs_important = [steps_change_occurs[i] for i in scaffold_points]

    # Add all scaffold labels
    for i, c in enumerate(steps_change_occurs):
        if c < steps_to_use-200:
            axs[3].vlines(c, 0, max(new_max_line_all[-1]), colors="orange", linestyles="dashed", linewidth=1)

    # Add main scaffold labels
    for i, c in enumerate(steps_change_occurs_important):
        axs[3].vlines(c, 0, max(new_max_line_all[-1]), colors="orange", linestyles="dashed", linewidth=3)
        axs[3].text(c, max(PCR_for_each[0])/10 * i, scaffold_point_names[i], fontsize=18)

    plt.tight_layout()
    plt.savefig("../Figures/Panels/Panel-3/Full-Metrics-Lines.png")
    plt.show()



model_list = ["dqn_scaffold_20-1", "dqn_scaffold_20-2", "dqn_scaffold_21-1", "dqn_scaffold_21-2"]
failed_list = ["dqn_no_scaffold-1", "dqn_no_scaffold-2"]
scaffold_points = [9, 13, 15]
scaffold_point_names = ["Shot noise Introduction", "Darken Arena", "Paramecia faster movement"]
plot_four_metrics_multiple_lines(model_list, scaffold_points, scaffold_point_names, 40)
plot_four_metrics_error_bars(model_list, scaffold_points, scaffold_point_names, 40)

# plot_all_metrics_multiple_models(model_list, scaffold_points, scaffold_point_names, 40)



# plot_pci_multiple_models(model_list, 30, scaffold_points, scaffold_point_names)
# plot_pai_multiple_models(model_list, 30, scaffold_points, scaffold_point_names)
# plot_pci_single("dqn_scaffold_20-1", 30)