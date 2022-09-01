import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import f_regression

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.label_behavioural_context import label_behavioural_context_multiple_trials,\
    get_behavioural_context_name_by_index
from Analysis.Behavioural.VisTools.get_action_name import get_action_name
from Analysis.Neural.Tools.normalise_activity import normalise_within_neuron

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


def build_full_regressor_set(datas, model_name):
    """For a single data object."""
    # One hot context variables
    labels = label_behavioural_context_multiple_trials(datas, model_name)
    label_names = [get_behavioural_context_name_by_index(i) for i in range(labels[0].shape[1])]

    # Continuous basic variables
    continuous_variables = ["energy_state", "salt", "salt_health"]# TODO: , "reward"]

    for i, data in enumerate(datas):
        # Continuous variables
        for var in continuous_variables:
            variables = np.expand_dims(data[var], 1)
            labels[i] = np.concatenate((labels[i], variables), axis=1)
            if i == 0:
                label_names.append(var)

        # One-hot Actions
        one_hot_encoded_actions, action_names = get_one_hot_encoded_actions_discrete(data)
        labels[i] = np.concatenate((labels[i], one_hot_encoded_actions), axis=1)

        if i == 0:
            label_names += action_names

    return labels, label_names


def build_regression_model_for_neuron(rnn_data, labels, cv_prop=0.1, ridge=True):
    """Does cross validated scoring."""

    # Specify model
    if ridge:
        regr = linear_model.Ridge(fit_intercept=False)
    else:
        regr = linear_model.LinearRegression(fit_intercept=False)

    # Separate Data
    cv_index = int(rnn_data.shape[0] * (1 - cv_prop))
    X = labels[:cv_index, :]
    Y = rnn_data[:cv_index]

    # Fit model
    regr.fit(X, Y)
    coefficients = regr.coef_

    # Compute f statistic
    f_statistic, p_value = f_regression(X, Y)
    f_statistic[np.isnan(f_statistic)] = 0

    score = regr.score(labels[cv_index:], rnn_data[cv_index:])

    return coefficients, score, f_statistic


def get_score_proportions(rnn_data, labels, cv_prop=0.1):
    cv_index = int(rnn_data.shape[0] * (1 - cv_prop))
    scores = []

    for i in range(labels.shape[1]):
        regr = linear_model.LinearRegression(fit_intercept=False)
        regr.fit(labels[:cv_index, i:i+1], rnn_data[:cv_index])
        score = regr.score(labels[cv_index:, i:i+1], rnn_data[cv_index:])
        scores.append(score)

    return scores

def normalise_continuous_regressors(regressors):
    continuous_indices = [12, 13]
    for i in continuous_indices:
        maxim, minim = np.max(regressors[:, i]), np.min(regressors[:, i])
        scaling = abs(maxim) + abs(minim)
        regressors[:, i] /= scaling
    return regressors


def build_all_regression_models(datas, model_name):
    labels, label_names = build_full_regressor_set(datas, model_name)

    labels = np.concatenate(labels, axis=0)
    labels = normalise_continuous_regressors(labels)

    compiled_coefficients = []
    scores = []
    relative_scores_compiled = []
    for n in range(512):

        compiled_rnn_data = datas[0]["rnn_state_actor"][:, 0, 0, n]
        for data in datas[1:]:
            compiled_rnn_data = np.concatenate((compiled_rnn_data, data["rnn_state_actor"][:, 0, 0, n]), axis=0)

        compiled_rnn_data = normalise_within_neuron(compiled_rnn_data)

        coefficient, score, relative_scores = build_regression_model_for_neuron(compiled_rnn_data, labels)

        compiled_coefficients.append(coefficient)
        scores.append(score)
        relative_scores_compiled.append(relative_scores)

    compiled_coefficients = np.array(compiled_coefficients)
    # compiled_coefficients = compiled_coefficients[:20]

    neg_scaling = abs(np.min(compiled_coefficients))
    positive_scaling = abs(np.max(compiled_coefficients))

    compiled_coefficients[compiled_coefficients > 0] /= positive_scaling
    compiled_coefficients[compiled_coefficients < 0] /= neg_scaling

    scores = np.array(scores)
    # scores = scores[:20]
    positive_scaling = abs(np.max(scores))
    scores /= positive_scaling

    coeff_and_scores = np.concatenate((np.expand_dims(scores, 1), compiled_coefficients), axis=1)

    # plt.imshow(coeff_and_scores, cmap="seismic")
    # plt.tight_layout()
    # plt.show()

    return compiled_coefficients, scores, relative_scores_compiled, label_names


def build_all_regression_models_activity_differential(datas, model_name):
    labels, label_names = build_full_regressor_set(datas, model_name)

    labels = [label[1:] for label in labels]
    labels = np.concatenate(labels, axis=0)
    labels = normalise_continuous_regressors(labels)

    compiled_coefficients = []
    scores = []
    relative_scores_compiled = []
    for n in range(512):

        compiled_rnn_data = datas[0]["rnn_state_actor"][1:, 0, 0, n] - datas[0]["rnn_state_actor"][:-1, 0, 0, n]
        for data in datas[1:]:
            compiled_rnn_data = np.concatenate((compiled_rnn_data,
                                                data["rnn_state_actor"][1:, 0, 0, n]-data["rnn_state_actor"][:-1, 0, 0, n]), axis=0)

        compiled_rnn_data = normalise_within_neuron(compiled_rnn_data)

        coefficient, score, relative_scores = build_regression_model_for_neuron(compiled_rnn_data, labels)

        compiled_coefficients.append(coefficient)
        scores.append(score)
        relative_scores_compiled.append(relative_scores)

    compiled_coefficients = np.array(compiled_coefficients)
    # compiled_coefficients = compiled_coefficients[:20]

    neg_scaling = abs(np.min(compiled_coefficients))
    positive_scaling = abs(np.max(compiled_coefficients))

    compiled_coefficients[compiled_coefficients > 0] /= positive_scaling
    compiled_coefficients[compiled_coefficients < 0] /= neg_scaling

    scores = np.array(scores)
    # scores = scores[:20]
    positive_scaling = abs(np.max(scores))
    scores /= positive_scaling

    coeff_and_scores = np.concatenate((np.expand_dims(scores, 1), compiled_coefficients), axis=1)

    # plt.imshow(coeff_and_scores, cmap="seismic")
    # plt.tight_layout()
    # plt.show()
    return compiled_coefficients, scores, relative_scores_compiled, label_names


if __name__ == "__main__":
    model_name = "dqn_scaffold_18-1"
    datas = []
    for i in range(1, 4):
        data = load_data(model_name, "Behavioural-Data-Endless", f"Naturalistic-{i}")
        datas.append(data)

    coefficients, scores, label_names, relative_scores = build_all_regression_models(datas, model_name)
    coefficients2, scores2, label_names2, relative_scores2 = build_all_regression_models_activity_differential(datas,
                                                                                                               model_name)
