import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter


from Analysis.load_data import load_data
from Analysis.Neural.Regression.build_regression_model import build_all_regression_models, \
    build_all_regression_models_activity_differential


def plot_category_distribution(neuron_categories):
    letter_counts = Counter(neuron_categories)
    df = pd.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar')
    plt.tight_layout()
    plt.show()


def categorise_neurons(coefficients, scores, labels, score_threshold):
    """Gives a neuron a role based on the encoded variable it most contributes to - the largest regressor."""
    coefficient_abs = np.absolute(coefficients)
    max_coefficient = np.argmax(coefficient_abs, axis=1)

    neuron_categories = []
    for c, s in zip(max_coefficient, scores):
        if s > score_threshold:
            neuron_categories.append(labels[c])
        else:
            neuron_categories.append("No Label Suitable")

    return neuron_categories


def get_neuron_indices_category(neuron_categories, chosen_category):
    indices = [i for i, n in enumerate(neuron_categories) if n == chosen_category]
    return indices


def get_category_indices(model_name, assay_config, assay_id, n, chosen_category, ignore_actions=True, score_threshold=0.2):
    datas = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        datas.append(data)

    # coefficients_1, scores_1, labels = build_all_regression_models(datas, model_name)
    # reduced_coefficients = coefficients_1[:, :15]
    # reduced_labels = labels[:15]
    # neuron_categories_1 = categorise_neurons(reduced_coefficients, scores_1, reduced_labels)

    coefficients_2, scores_2, labels = build_all_regression_models_activity_differential(datas, model_name)
    if ignore_actions:
        coefficients_2 = coefficients_2[:, :15]
        labels = labels[:15]
    neuron_categories_2 = categorise_neurons(coefficients_2, scores_2, labels, score_threshold)
    plot_category_distribution(neuron_categories_2)

    selected_neurons = get_neuron_indices_category(neuron_categories_2, chosen_category)
    return selected_neurons


if __name__ == "__main__":
    neurons = get_category_indices("dqn_scaffold_18-1", "Behavioural-Data-Endless", "Naturalistic", 3, "Starving")


