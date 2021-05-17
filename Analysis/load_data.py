import h5py
import numpy as np
import os


def load_data(model_name, assay_configuration, assay_id):
    """Loads the data of an individual assay from an assay configuration file."""
    file = h5py.File(f"../../Assay-Output/{model_name}/{assay_configuration}.h5", "r")
    g = file.get(assay_id)

    data = {key: np.array(g.get(key)) for key in g.keys()}
    file.close()
    return data
