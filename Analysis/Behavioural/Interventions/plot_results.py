import matplotlib.pyplot as plt
import seaborn as sns
from pylab import setp


from Analysis.Behavioural.Interventions.compare_behavioural_measures import get_both_measures, get_both_measures_random_gradient


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
    for ablation in ablations:
        plt.plot(percentages, ablation)
        average = [average[i] + abl for i, abl in enumerate(ablation)]
    for i, av in enumerate(average):
        average[i] = av/len(ablations)
    plt.plot(percentages, average, color="orange")
    plt.xlabel("Percentage ablated")
    plt.ylabel(measure)
    plt.legend()
    plt.show()

def plot_gradients(ablations, controls, ablation_group, measure):
    percentages = range(0, 110, 10)
    plt.plot(percentages, controls, label="Control (Random Ablations)")
    plt.plot(percentages, ablations, color="r", label="Targeted Ablations")
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


# show_gross_changes([22, 22], [30, 30], [30, 30], [10, 10], [10, 10], [10, 10])

# Random (all)
# prey_gradient_random1, pred_gradient_random1 = get_both_measures_random_gradient("new_even_prey_ref-1", "Indiscriminate-even_prey_only", 3)
# prey_gradient_random2, pred_gradient_random2 = get_both_measures_random_gradient("new_even_prey_ref-2", "Indiscriminate-even_prey_only", 3)
# prey_gradient_random3, pred_gradient_random3 = get_both_measures_random_gradient("new_even_prey_ref-2", "Indiscriminate-even_prey_only", 3)
# prey_gradient_random4, pred_gradient_random4 = get_both_measures_random_gradient("new_even_prey_ref-4", "Indiscriminate-even_prey_only", 3)

# prey_gradient_randomx, pred_gradient_random1 = get_both_measures_random_gradient("new_even_prey_ref-1", "Indiscriminate-even_naturalistic", 3)
# prey_gradient_randomx, pred_gradient_random2 = get_both_measures_random_gradient("new_even_prey_ref-2", "Indiscriminate-even_naturalistic", 3)
# prey_gradient_randomx, pred_gradient_random3 = get_both_measures_random_gradient("new_even_prey_ref-3", "Indiscriminate-even_naturalistic", 3)
# prey_gradient_randomx, pred_gradient_random4 = get_both_measures_random_gradient("new_even_prey_ref-4", "Indiscriminate-even_naturalistic", 3)


# plot_multiple_gradients_random([prey_gradient_random[:7] + [prey_gradient_random[7]] + prey_gradient_random[7:] for prey_gradient_random in [prey_gradient_random1,
# prey_gradient_random2,
# prey_gradient_random3,
# prey_gradient_random4]], "Prey Caught")
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

pred_gradient_random = [2, 2, 2, 2, 2, 2.66, 2.66, 2.66, 2.66, 2.66, 2.66] # Subsets for 13.6 percent
prey_gradient_random = [3, 3, 3, 3, 3, 2.66, 2.66, 2.66, 2.66, 2.66, 2.66]

#Prey-Large Central
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

# Predator-only
prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-4", "Predator_Only-even_predator", 3)
plot_gradients(prey_gradient, prey_gradient_random, "Predator-Only", "Prey Caught")
plot_gradients(pred_gradient, pred_gradient_random, "Predator-Only", "Predators Avoided")
print("\n")

prey_gradient, pred_gradient = get_both_measures("new_even_prey_ref-4", "Predator_Only-even_prey_only", 3)
plot_gradients(prey_gradient, prey_gradient_random, "Predator-Only", "Prey Caught")
plot_gradients(pred_gradient, pred_gradient_random, "Predator-Only", "Predators Avoided")


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

