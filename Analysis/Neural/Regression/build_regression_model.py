import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.label_behavioural_context import label_behavioural_context_multiple_trials,\
    get_behavioural_context_name_by_index
from Analysis.Behavioural.VisTools.get_action_name import get_action_name


"""
Building regression model.

Need to define variables to regress:
  - one hot encoding of some of the behavioural variables: prey capture, escape,
  - egocentric positioning of 


The model is then applied to all ANN units individually and fit created for them.

"""

def get_one_hot_encoded_actions_discrete(data):
    actions = [a for a in range(11)]

    for a in actions:

        timestamps = (data["action"] == a) * 1
        timestamps = np.expand_dims(timestamps, 1)
        if a == 0:
            compiled_actions = timestamps
        else:
            compiled_actions = np.concatenate((compiled_actions, timestamps), axis=1)

    actions = [get_action_name(a) for a in actions]

    return compiled_actions, actions

def build_full_regressor_set(data, model_name):
    """For a single data object."""
    # One hot context variables
    labels = label_behavioural_context_multiple_trials([data], model_name)[0]
    label_names = [get_behavioural_context_name_by_index(i) for i in range(labels.shape[1])]

    # Continuous basic variables
    continuous_variables = ["energy_state", "salt", "salt_health"]
    for var in continuous_variables:
        variables = np.expand_dims(data[var], 1)
        labels = np.concatenate((labels, variables), axis=1)
        label_names.append(var)

    # One-hot Actions
    one_hot_encoded_actions, action_names = get_one_hot_encoded_actions_discrete(data)
    labels = np.concatenate((labels, one_hot_encoded_actions), axis=1)
    label_names += action_names

    return labels, label_names


def build_regression_model_for_neuron(rnn_data, labels, label_names):
    # TODO: Need to find way of determining whether there is an association
    regr = linear_model.LinearRegression()
    regr.fit(labels, rnn_data)
    coefficients = regr.coef_
    encoded = (np.absolute(coefficients) > 1)
    print([label_names[i] for i, e in enumerate(encoded) if e])
    return coefficients


if __name__ == "__main__":
    model_name = "dqn_scaffold_18-1"
    data = load_data(model_name, "Behavioural-Data-Endless", "Naturalistic-2")
    labels, label_names = build_full_regressor_set(data, model_name)
    compiled_coefficients = []
    for n in range(512):
        rnn_data = data["rnn_state_actor"][:, 0, 0, n]
        coefficient = build_regression_model_for_neuron(rnn_data, labels, label_names)
        compiled_coefficients.append(coefficient)
    compiled_coefficients = np.array(compiled_coefficients)
    plt.imshow(compiled_coefficients)
    plt.tight_layout()
    # TODO: Find out why associations with actions end up being so high.
    plt.show()