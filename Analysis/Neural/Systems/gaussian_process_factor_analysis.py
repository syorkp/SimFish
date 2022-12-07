import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

from Analysis.load_data import load_data


def extract_traj(rnn_data):
    ...


if __name__ == "__main__":
    datas = []
    model_name = "dqn_scaffold_18-2"
    # for i in range(1, 2):
    #     data = load_data(model_name, "Behavioural-Data-Free", f"Naturalistic-{i}")
    for i in range(1, 2):
        data = load_data("dqn_scaffold_18-2", "Behavioural-Data-Free", f"Naturalistic-{i}")
        datas.append(data)
