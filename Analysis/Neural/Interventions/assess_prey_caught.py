import numpy as np

from Analysis.load_data import load_data


if __name__ == "__main__":
    for i in range(1, 4):
        ablation_data = load_data("dqn_scaffold_14-1", "Behavioural-Data-AblationsA", f"Naturalistic-{i}")
        print(np.sum(ablation_data["consumed"]))

    for i in range(1, 20):
        ablation_data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Free", f"Naturalistic-{i}")
        print(np.sum(ablation_data["consumed"]))
