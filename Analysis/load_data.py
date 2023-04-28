import h5py
import numpy as np
import os

import matplotlib.pyplot as plt


def load_data(model_name, assay_configuration, assay_id, training_data=False):
    """Loads the data of an individual assay from an assay configuration file."""
    # print(f"Attempting to load data from: {os.getcwd()}")

    if training_data:
        filepath = f"Training-Output/{model_name}/episodes/{assay_configuration}"
    else:
        filepath = f"Assay-Output/{model_name}/{assay_configuration}"

    try:
        file = h5py.File(f"../../../../{filepath}.h5", "r")
    except OSError:
        try:
            file = h5py.File(f"../../../{filepath}.h5", "r")
        except OSError:
            try:
                file = h5py.File(f"../../{filepath}.h5", "r")
            except OSError:
                try:
                    file = h5py.File(f"../{filepath}.h5", "r")
                except OSError:
                    file = h5py.File(f"./{filepath}.h5", "r")

    g = file.get(assay_id)

    data = {key: np.array(g.get(key)) for key in g.keys()}
    file.close()
    return data


if __name__ == "__main__":
    # d = load_data("dqn_new_pred-1", "Episode 1", f"Episode 1", training_data=True)
    #
    # fish_prey_vectors = np.expand_dims(d["fish_position"], 1) - d["prey_positions"]
    # fish_prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    # min_distance = np.min(fish_prey_distances, axis=1)
    # max_uv = np.max(d["observation"][:, :, 1, :], axis=(1, 2))
    # comb = np.concatenate((np.expand_dims(min_distance, 1), np.expand_dims(max_uv, 1)), axis=1)
    d = load_data("dqn_0_0-1", "Controlled-Visual-Stimuli", f"Moving-Predator-1")

    d = load_data("dqn_0_0-1", "Behavioural-Data-Free", f"Naturalistic-1")
    d2 = load_data("dqn_0_0-1", "Behavioural-Data-Free", f"Naturalistic-Mod-1")

    d = load_data("dqn_salt_only_reduced-1", "Episode 900", f"Episode 900", training_data=True)
    print(f"1-{np.sum(d['action'] == 1)}")
    print(f"2-{np.sum(d['action'] == 2)}")
    print(f"4-{np.sum(d['action'] == 4)}")
    print(f"5-{np.sum(d['action'] == 5)}")
    print(f"10-{np.sum(d['action'] == 10)}")
    print(f"11-{np.sum(d['action'] == 11)}")
    # datas = []
    # obs = []
    for i in range(1, 7):
        d = load_data("dqn_0-1", "Behavioural-Data-Free", f"Naturalistic-{i}", training_data=False)
        print(f"{i}-{np.sum(d['consumed'])}")
    #     print(f"2-{np.sum(d['action'] == 2)}")
    #     print(f"4-{np.sum(d['action'] == 4)}")
    #     print(f"5-{np.sum(d['action'] == 5)}")
    #     print(f"10-{np.sum(d['action'] == 10)}")
    #     print(f"11-{np.sum(d['action'] == 11)}")
    # obs = np.concatenate(obs)
    # red = obs[:, :, 0, :].flatten()
    # uv = obs[:, :, 1, :].flatten()
    # red2 = obs[:, :, 2, :].flatten()
    #
    # plt.hist(red)
    # plt.show()
    #
