from Analysis.load_data import load_data
from Analysis.Behavioural.New.display_action_sequences import display_all_sequences_capture, get_capture_sequences
from Analysis.Behavioural.Interventions.compare_behavioural_measures import get_both_measures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extract_attempted_captures(data):
    prey_timestamps = []
    sensing_distance = 60
    for i, p in enumerate(data["position"]):
        for prey in data["prey_positions"][i]:
            sensing_area = [[p[0] - sensing_distance,
                             p[0] + sensing_distance],
                            [p[1] - sensing_distance,
                             p[1] + sensing_distance]]
            near_prey = sensing_area[0][0] <= prey[0] <= sensing_area[0][1] and \
                        sensing_area[1][0] <= prey[1] <= sensing_area[1][1]
            if near_prey:
                prey_timestamps.append(i)
                break
    successful_captures = sum(data["consumed"])
    attempted_captures = 0
    last_index = 0
    for i in prey_timestamps:
        if last_index == 0:
            last_index = i
            continue
        if i-1 == last_index:
            last_index = i
        else:
            last_index = i
            attempted_captures += 1
    return attempted_captures, successful_captures


def get_all_attempted_captures(model_name, assay_name, n=6):
    percentages = range(0, 110, 10)
    attempted_captures = []
    successful_captures = []
    sns.set()
    for per in percentages:
        attempted, successful = 0, 0
        for i in range(1, n+1):
            ac, sc = extract_attempted_captures(load_data(model_name, f"Ablation-Test-{assay_name}", f"Ablated-{per}-{i}"))
            attempted += ac
            successful += sc
        attempted_captures.append(attempted/n)
        successful_captures.append(successful/n)
    return attempted_captures


def plot_attempted_captures(model_name, assay_name, n=6):
    percentages = range(0, 110, 10)
    attempted_captures = []
    successful_captures = []
    sns.set()
    for per in percentages:
        attempted, successful = 0, 0
        for i in range(1, n+1):
            ac, sc = extract_attempted_captures(load_data(model_name, f"Ablation-Test-{assay_name}", f"Ablated-{per}-{i}"))
            attempted += ac
            successful += sc
        attempted_captures.append(attempted/n)
        successful_captures.append(successful/n)

    plt.plot(percentages, successful_captures, label="Successful captures")
    plt.plot(percentages, attempted_captures, label="Attempted captures")
    plt.xlabel("Percentage ablated")
    plt.ylabel("Captures")
    plt.legend()
    plt.show()


plot_attempted_captures("even_prey_ref-4", "15-in-front")
