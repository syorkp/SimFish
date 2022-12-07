import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def plot_output_and_hidden_activity(rnn_data):
    # plt.plot(rnn_data[:, 0])
    plt.plot(rnn_data[:, 1])
    plt.show()


if __name__ == "__main__":
    data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
    for i in range(10):
        rnn_activity = data["rnn_state_actor"][:, :, 0, i]
        plot_output_and_hidden_activity(rnn_activity)
