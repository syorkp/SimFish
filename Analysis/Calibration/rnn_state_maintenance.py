import matplotlib.pyplot as plt

import json
import numpy as np

def load_states_data(model_name, episode_number):
    with open(f"../../Training-Output/{model_name}/rnn_state-{episode_number}.json", 'r') as f:
        data = json.load(f)
        num_rnns = len(data.keys()) / 4
        init_rnn_state = tuple(
            (np.array(data[f"rnn_state_{shape}_1"]), np.array(data[f"rnn_state_{shape}_2"])) for shape in
            range(int(num_rnns)))
        init_rnn_state_ref = tuple(
            (np.array(data[f"rnn_state_{shape}_ref_1"]), np.array(data[f"rnn_state_{shape}_ref_2"])) for shape in
            range(int(num_rnns)))
    return init_rnn_state


def plot_rnn_activity_across_training(rnn_state_data):
    relevant_data_points = [r[0][0][0] for r in rnn_state_data]
    relevant_data_points = np.array(relevant_data_points)
    relevant_data_points = np.swapaxes(relevant_data_points, 0, 1)

    plt.imshow(relevant_data_points, aspect="auto")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    rnn_state_data = [load_states_data("dqn_scaffold_beta_test-1", i) for i in range(0, 480, 20)]
    plot_rnn_activity_across_training(rnn_state_data)


