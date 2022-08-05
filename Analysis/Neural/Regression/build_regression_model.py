import numpy as np
from sklearn import linear_model

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.label_behavioural_context import label_behavioural_context_multiple_trials,\
    get_behavioural_context_name_by_index

"""
Building regression model.

Need to define variables to regress:
  - one hot encoding of some of the behavioural variables: prey capture, escape,
  - egocentric positioning of 


The model is then applied to all ANN units individually and fit created for them.

"""

if __name__ == "__main__":
    data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Endless", "Naturalistic-2")
    labels = label_behavioural_context_multiple_trials([data], "dqn_scaffold_18-1")[0]
    label_names = [get_behavioural_context_name_by_index(i) for i in range(labels.shape[1])]
    for n in range(2):
        regr = linear_model.LinearRegression()
        regr.fit(labels, data["rnn_state_actor"][:, 0, 0, n])
        coefficients = regr.coef_
        encoded = (np.absolute(coefficients) > 1)
        print([label_names[i] for i, e in enumerate(encoded) if e])

