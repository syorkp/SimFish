import h5py
import numpy as np

# Note that RNN activity data is arange in the format - (unit, 1, timestep), so indexing properly is important.

def load_data(configuration_name, assay_id):
    """Loads the data of an individual assay from an assay configuration file."""

    file = h5py.File(f"../Assay-Output/base-1/{configuration_name}.h5", "r")
    g = file.get(assay_id)

    data = {key: np.array(g.get(key)) for key in g.keys()}
    print(data.keys())
    return data




