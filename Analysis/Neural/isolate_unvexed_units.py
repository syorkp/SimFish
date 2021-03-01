from Analysis.load_data import load_data
from Analysis.Visualisation.display_many_neurons import plot_multiple_traces


# def isolate_unvexed_units(data, unchaning_period=1000):
#     for unit in data:
#


data = load_data("large_all_features-1", "No_Stimuli", "No_Stimuli")

unit_activity = [[data["rnn state"][i - 1][0][j] for i in data["step"]] for j in range(512)]
plot_multiple_traces(unit_activity)

