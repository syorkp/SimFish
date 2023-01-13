import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def plot_consumption_rates_over_time(model_name, assay_config, assay_id, n, window_size=100):
    consumption_step_count = np.zeros((1000))
    min_length = 1000#000
    for trial in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{trial}")
        consumptions = data["consumed"] * 1
        if consumptions.shape[0] < min_length:
            min_length = consumptions.shape[0]

        if consumptions.shape[0] > consumption_step_count.shape[0]:
            consumption_step_count = np.concatenate((consumption_step_count, np.zeros((consumptions.shape[0]-consumption_step_count.shape[0]))))
        elif consumptions.shape[0] < consumption_step_count.shape[0]:
            consumptions = np.concatenate(
                (consumptions, np.zeros((consumption_step_count.shape[0] - consumptions.shape[0]))))

        consumption_step_count += consumptions

    consumption_step_count = consumption_step_count[:min_length]
    consumption_step_count /= n

    consumption_rate = np.array([np.sum(consumption_step_count[i:i+window_size]) for i in range(consumption_step_count.shape[0]-window_size)])

    plt.plot(consumption_rate)
    plt.show()



if __name__ == "__main__":
    plot_consumption_rates_over_time("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20, window_size=20)

