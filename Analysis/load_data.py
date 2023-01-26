import h5py
import numpy as np
import os


def load_data(model_name, assay_configuration, assay_id):
    """Loads the data of an individual assay from an assay configuration file."""
    print(f"Attempting to load data from: {os.getcwd()}")

    try:
        file = h5py.File(f"../../../../Assay-Output/{model_name}/{assay_configuration}.h5", "r")
    except OSError:
        try:
            file = h5py.File(f"../../../Assay-Output/{model_name}/{assay_configuration}.h5", "r")
        except OSError:
            try:
                file = h5py.File(f"../../Assay-Output/{model_name}/{assay_configuration}.h5", "r")
            except OSError:
                try:
                    file = h5py.File(f"../Assay-Output/{model_name}/{assay_configuration}.h5", "r")
                except OSError:
                    file = h5py.File(f"./Assay-Output/{model_name}/{assay_configuration}.h5", "r")

    g = file.get(assay_id)
    # print(assay_id)
    data = {key: np.array(g.get(key)) for key in g.keys()}
    file.close()
    return data


if __name__ == "__main__":
    datas = []
    d = load_data("dqn_gamma_pm-2", "Behavioural-Data-Free", f"Naturalistic-2")
    actions = d["action"]
    prey_positions = d["prey_positions"]
    prey_velocity = prey_positions[1:] - prey_positions[:-1]
    prey_velocity *= 5/10
    prey_velocity2 = (prey_velocity[:, :, 0] ** 2 + prey_velocity[:, :, 1] ** 2) ** 0.5
    # for i in range(1, 21):
    #     d = load_data("dqn_beta-1", "Behavioural-Data-Free", f"Naturalistic-{i}")
    #     print(f"{i}-{np.sum(d['consumed'] * 1)}")
    #     datas.append(d)
