import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def show_energy_state(data, datapoint_index):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(data["energy_state"])

    et = data["energy_state"][datapoint_index]
    consumption_timepoints = [i for i, c in enumerate(data["consumed"]) if c == 1]

    for c in consumption_timepoints:
        ax.vlines(c, 0, data["energy_state"][c], color="green")

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.hlines(et, 0, datapoint_index, colors="r")
    ax.vlines(datapoint_index, 0, et, colors="r")

    ax.margins(x=0, y=0)
    ax.legend(["Energy State", "Consumption"], fontsize=14)
    plt.ylabel("Energy State", size=25)
    plt.xlabel("Step", size=25)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    plt.tight_layout()
    plt.savefig(f"./Panels/Panel-2/energy_state.jpg")

    plt.show()


dat = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-4")
show_energy_state(dat, 550)
