import h5py
import numpy as np
import os


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
    datas = []
    obs = []
    for i in range(1, 11):
        d = load_data("dqn_epsilon-6", "Behavioural-Data-Free", f"Naturalistic-{i}", training_data=False)
        obs.append(d["observation"])
    obs = np.concatenate(obs)
    red = obs[:, :, 0, :].flatten()
    uv = obs[:, :, 1, :].flatten()
    red2 = obs[:, :, 2, :].flatten()

    import matplotlib.pyplot as plt
    plt.hist(uv)
    plt.show()

