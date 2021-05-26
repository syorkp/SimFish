import matplotlib.pyplot as plt
import seaborn as sns
from pylab import setp
import numpy as np

from Analysis.Behavioural.Interventions.compare_behavioural_measures import get_both_measures, \
    get_both_measures_random_gradient


def plot_gradients_random(ablations, measure):
    percentages = range(0, 105, 5)
    plt.plot(percentages, ablations)
    plt.xlabel("Percentage ablated")
    plt.ylabel(measure)
    plt.legend()
    plt.show()


def plot_multiple_gradients_random(ablations, measure):
    percentages = range(0, 105, 5)
    sns.set()
    average = [0 for i in percentages]
    std = [np.std([a[i] for a in ablations]) for i, point in enumerate(ablations[0])]
    for ablation in ablations:
        # plt.plot(percentages, ablation)
        average = [average[i] + abl for i, abl in enumerate(ablation)]
    for i, av in enumerate(average):
        average[i] = av / len(ablations)
    plt.plot(percentages, average, color="orange")
    stdlow = [a - s for a, s in zip(average, std)]
    stdhigh = [a + s for a, s in zip(average, std)]

    high_v = [max([a[i] for a in ablations]) for i, point in enumerate(ablations[0])]
    low_v = [min([a[i] for a in ablations]) for i, point in enumerate(ablations[0])]

    # plt.fill_between(percentages, stdlow, stdhigh)

    plt.fill_between(percentages, low_v, high_v)
    plt.hlines(1, 0, 100, color="r")
    plt.xlabel("Percentage ablated")
    plt.ylabel(measure)
    # plt.legend()
    plt.show()


def plot_multiple_gradients(ablations, controls, ablation_group, measure):
    percentages = range(0, 110, 10)
    sns.set()
    average = [0 for i in percentages]
    std = [np.std([a[i] for a in ablations]) for i, point in enumerate(ablations[0])]
    for ablation in ablations:
        # plt.plot(percentages, ablation)
        average = [average[i] + abl for i, abl in enumerate(ablation)]
    for i, av in enumerate(average):
        average[i] = av / len(ablations)
    plt.plot(percentages, average, color="orange", label=ablation_group)
    stdlow = [a - s for a, s in zip(average, std)]
    stdhigh = [a + s for a, s in zip(average, std)]

    high_v = [max([a[i] for a in ablations]) for i, point in enumerate(ablations[0])]
    low_v = [min([a[i] for a in ablations]) for i, point in enumerate(ablations[0])]

    plt.fill_between(percentages, low_v, high_v)
    plt.plot(percentages, controls, label="Control (Random Ablations)", color="r")
    plt.plot(percentages, [1 for i in percentages], label="Baseline Predator Avoidance", color="g")
    plt.xlabel("Percentage ablated")
    plt.legend()

    plt.ylabel(measure)
    # plt.legend()
    plt.show()


def plot_gradients(ablations, controls, ablation_group, measure):
    percentages = range(0, 110, 10)
    plt.plot(percentages, controls, label="Control (Random Ablations)")
    plt.plot(percentages, ablations, color="r", label=ablation_group)
    plt.xlabel("Percentage ablated")
    plt.ylabel(measure)
    plt.legend()
    plt.title(f"Ablations in {ablation_group} Neurons")
    # plt.ylim([0, max(controls) + 2])
    plt.show()


def setBoxColors(bp):
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], color='red')
    setp(bp['caps'][2], color='red')
    setp(bp['caps'][3], color='red')
    setp(bp['whiskers'][2], color='red')
    setp(bp['whiskers'][3], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][1], color='red')


def show_gross_changes(full_ablations_prey, control_ablations_prey, no_ablations_prey, full_ablations_pred,
                       control_ablations_pred, no_ablations_pred):
    bp = plt.boxplot([full_ablations_prey, control_ablations_prey, no_ablations_prey])
    plt.show()

    bp = plt.boxplot([full_ablations_pred, control_ablations_pred, no_ablations_pred])

    plt.show()


def get_random_control_average(gradients, dimensions, max_percentage):
    average = [0 for i in gradients[0]]
    std = [np.std([a[i] for a in gradients]) for i, point in enumerate(gradients[0])]
    for ablation in gradients:
        # plt.plot(percentages, ablation)
        average = [average[i] + abl for i, abl in enumerate(ablation)]
    for i, av in enumerate(average):
        average[i] = av / len(gradients)
    interpolated_indexes = np.linspace(0, max_percentage / 5, dimensions)
    interpolated_indexes = [int(i) for i in interpolated_indexes]
    return [average[i] if i < len(average) else average[i - 1] for i in interpolated_indexes]


# show_gross_changes([22, 22], [30, 30], [30, 30], [10, 10], [10, 10], [10, 10])

# Random (all)
# prey_gradient_random1, pred_gradient_random1 = get_both_measures_random_gradient("new_even_prey_ref-1",
#                                                                                  "Indiscriminate-even_prey_only", 3)
# prey_gradient_random2, pred_gradient_random2 = get_both_measures_random_gradient("new_even_prey_ref-2",
#                                                                                  "Indiscriminate-even_prey_only", 3)
# prey_gradient_random3, pred_gradient_random3 = get_both_measures_random_gradient("new_even_prey_ref-2",
#                                                                                  "Indiscriminate-even_prey_only", 3)
# prey_gradient_random4, pred_gradient_random4 = get_both_measures_random_gradient("new_even_prey_ref-4",
#                                                                                  "Indiscriminate-even_prey_only", 3)

# random_prey_gradient = get_random_control_average([prey_gradient_random1,
#                                                    prey_gradient_random2,
#                                                    prey_gradient_random3,
#                                                    prey_gradient_random4], 11, 10)

prey_gradient_randomx, pred_gradient_random1 = get_both_measures_random_gradient("new_even_prey_ref-4",
                                                                                 "Indiscriminate-even_predator", 6)
prey_gradient_randomx, pred_gradient_random2 = get_both_measures_random_gradient("new_even_prey_ref-5",
                                                                                 "Indiscriminate-even_predator", 6)
prey_gradient_randomx, pred_gradient_random3 = get_both_measures_random_gradient("new_even_prey_ref-5",
                                                                                 "Indiscriminate-even_predator", 6)
prey_gradient_randomx, pred_gradient_random4 = get_both_measures_random_gradient("new_even_prey_ref-8",
                                                                                 "Indiscriminate-even_predator", 6)

plot_multiple_gradients_random([pred_gradient_random1,
                                 pred_gradient_random2,
                                 pred_gradient_random3,
                                 pred_gradient_random4], "Predators Avoided")

x = True

#
# plot_multiple_gradients_random([pred_gradient_random[:7] + [pred_gradient_random[7]] + pred_gradient_random[7:] for pred_gradient_random in [pred_gradient_random1,
# pred_gradient_random2,
# pred_gradient_random3,
# pred_gradient_random4]], "Prey Caught")


# # plot_gradients_random(prey_gradient_random[:7] + [prey_gradient_random[7]] + prey_gradient_random[7:], "Prey Caught")
# print("\n")
#
# prey_gradient_random, pred_gradient_random = get_both_measures_random_gradient("new_even_prey_ref-1", "Indiscriminate-even_prey_only", 3)
# plot_gradients_random(prey_gradient_random[:7] + [prey_gradient_random[7]] + prey_gradient_random[7:], "Prey Caught")
# plot_gradients_random(pred_gradient_random[:7] + [pred_gradient_random[7]] + pred_gradient_random[7:], "Predators Avoided")

pred_random_gradient = [3 for i in range(11)]
# Prey-Large Central
prey_gradient1, pred_gradient = get_both_measures("new_even_prey_ref-1", "Prey-Large-Central-even_prey_only", 3)
prey_gradient2, pred_gradient = get_both_measures("new_even_prey_ref-2", "Prey-Large-Central-even_prey_only", 3)
prey_gradient3, pred_gradient = get_both_measures("new_even_prey_ref-3", "Prey-Large-Central-even_prey_only", 3)
prey_gradient4, pred_gradient = get_both_measures("new_even_prey_ref-4", "Prey-Large-Central-even_prey_only", 3)
#
# plot_multiple_gradients([prey_gradient1,
#                          prey_gradient2,
#                          prey_gradient3,
#                          prey_gradient4], random_prey_gradient, "Prey-in-Front", "Prey Caught")

prey_gradient1, pred_gradient1 = get_both_measures("new_even_prey_ref-4", "Prey-Large-Central-even_naturalistic", 3)
prey_gradient2, pred_gradient2 = get_both_measures("new_even_prey_ref-2", "Prey-Large-Central-even_naturalistic", 3)
prey_gradient3, pred_gradient3 = get_both_measures("new_even_prey_ref-3", "Prey-Large-Central-even_naturalistic", 3)
prey_gradient4, pred_gradient4 = get_both_measures("new_even_prey_ref-4", "Prey-Large-Central-even_naturalistic", 3)

plot_multiple_gradients([pred_gradient1,
                         pred_gradient2,
                         pred_gradient3,
                         pred_gradient4], pred_random_gradient, "Prey-in-Front", "Predators Avoided")

# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-3", "Prey-Large-Central-even_prey_only", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Prey-Large-Central", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Prey-Large-Central", "Predators Avoided")
# print("\n")
#
# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-3", "Prey-Large-Central-even_naturalistic", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Prey-Large-Central", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Prey-Large-Central", "Predators Avoided")

# Exploration
# prey_gradient, pred_gradient = get_both_measures("new_differential_prey_ref-4", "Exploration-differential_naturalistic", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Prey-Large-Central", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Prey-Large-Central", "Predators Avoided")
# print("\n")
#
# prey_gradient, pred_gradient = get_both_measures("new_differential_prey_ref-4", "Exploration-differential_prey_low_predator", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Prey-Large-Central", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Prey-Large-Central", "Predators Avoided")

# # Predator-only
# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-4", "Predator_Only-even_predator", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Predator-Only", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Predator-Only", "Predators Avoided")
# print("\n")
#
# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-4", "Predator_Only-even_prey_only", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Predator-Only", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Predator-Only", "Predators Avoided")


# # Predator-selective
# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-6", "Predator_Selective-even_predator", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Predator-Selective", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Predator-Selective", "Predators Avoided")
# print("\n")
# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-6", "Predator_Selective-even_prey_only", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Predator-Selective", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Predator-Selective", "Predators Avoided")

# Full field
# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-4", "Prey-Full-Field-even_prey_only", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Prey-Full-Field", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Prey-Full-Field", "Predators Avoided")
# print("\n")
#
# prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-4", "Prey-Full-Field-even_naturalistic", 3)
# plot_gradients(prey_gradient, prey_gradient_random, "Prey-Full-Field", "Prey Caught")
# plot_gradients(pred_gradient, pred_gradient_random, "Prey-Full-Field", "Predators Avoided")
