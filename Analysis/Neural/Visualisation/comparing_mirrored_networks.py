import matplotlib.pyplot as plt


from Analysis.load_data import load_data


if __name__ == "__main__":
    d = load_data("dqn_gamma-3", "Behavioural-Data-Free", "Naturalistic-1")
    # 2, 3, 4, 5, 7
    plt.plot(d["rnn_state_actor"][:, 0, 5])

    plt.plot(d["rnn_state_actor_ref"][:, 0, 5])
    plt.show()

