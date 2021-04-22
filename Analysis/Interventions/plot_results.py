import matplotlib.pyplot as plt
import seaborn as sns
from pylab import boxplot, setp


from Analysis.Interventions.compare_behavioural_measures import get_both_measures


def plot_gradients(ablations, controls, ablation_group, measure):
    percentages = range(0, 110, 10)
    plt.plot(percentages, controls, label="Control (Random Ablations)")
    plt.plot(percentages, ablations, color="r", label="Targeted Ablations")
    plt.xlabel("Percentage ablated")
    plt.ylabel(measure)
    plt.legend()
    plt.title(f"Ablations in {ablation_group} Neurons")
    plt.ylim([0, max(controls) + 2])
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


predator_control = [23, 22, 20.5, 21, 20, 21, 21, 21, 20, 20]
pre_control = [23, 23, 23, 22, 23, 21, 23, 22, 22, 23]
preon_control = [23, 22, 20.5, 20, 22, 21, 21, 20, 19, 20]
control = pre_control

gradient_results, control_abl, control_no_abl = get_both_measures("even_prey_ref-7", "Prey-in-Front", 3)

plot_gradients([control_no_abl] + gradient_results, [control_no_abl]+control, "Prey-in-Front", "Prey Caught")

# show_gross_changes([22, 22], [30, 30], [30, 30], [10, 10], [10, 10], [10, 10])