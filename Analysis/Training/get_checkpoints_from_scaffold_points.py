from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np


from Analysis.Training.load_from_logfiles import load_all_log_data

"""
Utility script to find the checkpoint numbers for selecting a given scaffold point.

This will be useful for gathering data at different stages in scaffold.
"""


def get_checkpoint(model_name, scaffold_point):
    scaffold_point_transition = scaffold_point + 1

    scaffold_point_changes = load_all_log_data(model_name)["Configuration change"]
    scaffold_point_changes = np.array(scaffold_point_changes).astype(int)
    scaffold_start_eps = scaffold_point_changes[(scaffold_point_changes[:, 1] == scaffold_point), 0] # [0]
    scaffold_end_eps = scaffold_point_changes[(scaffold_point_changes[:, 1] == scaffold_point_transition), 0] # [0]

    if len(scaffold_start_eps) == 0:
        print("Scaffold point not yet reached")
        return

    if len(scaffold_end_eps) == 0:
        print("Scaffold point not finished yet")
        return

    scaffold_start_eps = scaffold_start_eps[0]
    scaffold_end_eps = scaffold_end_eps[0]

    model_location = f"../../Training-Output/{model_name}"

    model_checkpoint_indexes = [f for f in listdir(model_location) if isfile(join(model_location, f)) and ".index" in f]
    checkpoint_nums = [int(checkpoint_path.split("model-")[-1][:-11]) for checkpoint_path in model_checkpoint_indexes]
    checkpoint_nums = sorted(checkpoint_nums)

    best_checkpoint = None

    for ep in range(scaffold_start_eps + 1, scaffold_end_eps + 1):
        if ep in checkpoint_nums:
            best_checkpoint = ep


    # Alert if there is no checkpoint within the scaffold point...
    if best_checkpoint is None:
        print("Error, no checkpoint available within scaffold point")

    return best_checkpoint


if __name__ == "__main__":
    c1 = get_checkpoint("dqn_gamma-1", 44)
    c2 = get_checkpoint("dqn_gamma-2", 44)
    c3 = get_checkpoint("dqn_gamma-3", 44)
    c4 = get_checkpoint("dqn_gamma-4", 44)
    c5 = get_checkpoint("dqn_gamma-5", 44)
