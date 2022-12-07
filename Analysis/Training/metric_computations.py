import numpy as np


def compute_one_hot_metric(data, death_type, window):
    if death_type is None:
        death_int = 0
    elif death_type == "Predation":
        death_int = 1
    elif death_type == "Prey-All-Eaten":
        death_int = 2
    elif death_type == "Starvation":
        death_int = 3
    else:
        print("Cause of death label wrong")
        return

    metric_array = []
    events = (data == death_int) * 1

    for i in range(len(data)-window):
        metric = np.sum(events[i: i+window])/window
        metric_array.append(metric)

    return metric_array


if __name__ == "__main__":
    m = compute_one_hot_metric(np.array([1, 0, 0, 0, 0, 1, 2, 3, 0]), "Starvation", 2)







