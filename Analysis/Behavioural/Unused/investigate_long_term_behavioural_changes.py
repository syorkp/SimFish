from scipy import stats
import matplotlib.pyplot as plt

from Analysis.load_data import load_data


def get_modal_action(data, binsize=5):
    modal_actions = []
    for i, action in enumerate(data["behavioural choice"]):
        if i % binsize == 0:
            subset = data["behavioural choice"][i:i + binsize]
            modal_actions.append(stats.mode(subset)[0])
    return modal_actions


data = load_data("large_all_features-1", "No_Stimuli", "No_Stimuli")

ma = get_modal_action(data)
plt.plot(ma)
plt.show()

x = True
