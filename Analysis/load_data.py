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
    d = load_data("dqn_epsilon-6", "Episode 200", "Episode 200", training_data=True)

    fig, axs = plt.subplots(3, sharex=True)
    axs[0].hist(d["observation"][:, :, 0].flatten())
    axs[1].hist(d["observation"][:, :, 1].flatten())
    axs[2].hist(d["observation"][:, :, 2].flatten())
    plt.show()
    datas = []
    # red_channel = []
    # for i in range(1, 101):
    #     d = load_data("dqn_gamma-4", "Behavioural-Data-Free-C", f"Naturalistic-{i}")
    #     print(f'{i} - {np.sum(d["consumed"])}')

