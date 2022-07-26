import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.label_behavioural_context import label_behavioural_context_multiple_trials


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


def plot_pca_trajectory_with_contexts_multiple_trials(datas):
    behavioural_labels = label_behavioural_context_multiple_trials(datas, environment_size=1500)
    all_activity_data = [np.swapaxes(data["rnn_state_actor"][:, 0, 0, :], 0, 1) for data in datas]
    plot_pca_trajectory_with_context(all_activity_data, behavioural_labels)


def plot_pca_trajectory_with_context(activity_data, associated_periods):
    flattened_activity_data = np.concatenate((activity_data), axis=1)
    pca = PCA(n_components=2)
    pca.fit(flattened_activity_data)
    pca_components = pca.components_

    split_colours = np.array([])
    for i in range(len(activity_data)):
        split_colours = np.concatenate((split_colours, np.arange(len(activity_data[i][0]))))

    prevailing_context_full = []
    for trial in associated_periods:
        for step in trial:
            if 3 in step:
                prevailing_context_full.append("^")
            else:
                if 1 in step:
                    prevailing_context_full.append("1")
                else:
                    if 3 in step or 4 in step:
                        prevailing_context_full.append("s")
                    else:
                        prevailing_context_full.append(".")

    plt.rcParams["figure.figsize"] = (20, 20)
    mscatter(pca_components[0], pca_components[1], c=split_colours, m=prevailing_context_full)
    plt.show()


if __name__ == "__main__":
    datas = []
    for i in range(1, 11):
        data = load_data("dqn_scaffold_18-2", "Behavioural-Data-Free", f"Naturalistic-{i}")
        datas.append(data)
    plot_pca_trajectory_with_contexts_multiple_trials(datas)






