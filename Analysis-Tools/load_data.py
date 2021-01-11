import h5py
import numpy as np


def load_data(configuration_name, assay_id, model_name="base-1"):
    """Loads the data of an individual assay from an assay configuration file."""

    file = h5py.File(f"../Assay-Output/{model_name}/{configuration_name}.h5", "r")
    g = file.get(assay_id)

    data = {key: np.array(g.get(key)) for key in g.keys()}
    file.close()
    return data




