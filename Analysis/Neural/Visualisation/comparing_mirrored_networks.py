import matplotlib.pyplot as plt


from Analysis.load_data import load_data


if __name__ == "__main__":
    d = load_data("dqn_new-1", "Episode 1", "Episode 1", training_data=True)
    # 2, 3, 4, 5, 7
    unit = 10
    plt.plot(d["rnn_state"][:, 0, unit])

    plt.plot(d["rnn_state_ref"][:, 0, unit])
    plt.show()

