from Analysis.load_data import load_data


def get_predator_num(data):
    tally = 0
    sequence=False
    for p in data['predator']:
        if not sequence and p:
            sequence = True
            tally += 1
        elif not p:
            sequence = False
    if len(data['consumed']) < 1000:
        tally = tally - 1
    return tally


def get_measures_targeted(model, targeted_neurons, percentage, number_of_trials):
    prey_caught = 0
    predators_avoided = 0
    for i in range(1, number_of_trials+1):
        print(percentage)
        data1 = load_data(model, f"Ablation-Test-{targeted_neurons}", f"Ablated-{percentage}-{i}")
        prey_caught = prey_caught + sum(data1['consumed'])
        predators_avoided = predators_avoided + get_predator_num(data1)
    return prey_caught/number_of_trials, predators_avoided/number_of_trials


def get_measures_random_gradient(model, targeted_neurons, percentage, number_of_trials):
    prey_caught = 0
    predators_avoided = 0
    for i in range(1, number_of_trials+1):
        data1 = load_data(model, f"Ablation-{targeted_neurons}", f"Ablated-{percentage}-{i}")
        prey_caught = prey_caught + sum(data1['consumed'])
        predators_avoided = predators_avoided + get_predator_num(data1)
    return prey_caught/number_of_trials, predators_avoided/number_of_trials


def get_measures_random(model, targeted_neurons, number_of_trials):
    prey_caught = 0
    predators_avoided = 0
    for i in range(1, number_of_trials+1):
        data1 = load_data(model, f"Ablation-Test-{targeted_neurons}", f"Random-Control-{i}")
        prey_caught = prey_caught + sum(data1['consumed'])
        predators_avoided = predators_avoided + get_predator_num(data1)
    return prey_caught/number_of_trials, predators_avoided/number_of_trials


def get_both_measures(model, targeted_neurons, number_of_trials):
    prey_caught_gradient = []
    predators_avoided_gradient = []
    for per in range(0, 110, 10):
        prey_caught, predators_avoided = get_measures_targeted(model, targeted_neurons, per, number_of_trials)
        print(f"Ablations: {per}%, Prey caught: {prey_caught}, Predators avoided: {predators_avoided}")
        prey_caught_gradient.append(prey_caught)
        predators_avoided_gradient.append(predators_avoided)

    prey_caught_cont_abl, predators_avoided = get_measures_random(model, targeted_neurons, number_of_trials)
    print(f"Control (random ablation): Prey caught: {prey_caught_cont_abl}, Predators avoided: {predators_avoided}")
    return prey_caught_gradient, predators_avoided_gradient


def get_both_measures_random_gradient(model, targeted_neurons, number_of_trials):
    prey_caught_gradient = []
    predators_avoided_gradient = []
    for per in range(0, 105, 5):
        if per == 35: continue
        prey_caught, predators_avoided = get_measures_random_gradient(model, targeted_neurons, per, number_of_trials)
        print(f"Ablations: {per}%, Prey caught: {prey_caught}, Predators avoided: {predators_avoided}")
        prey_caught_gradient.append(prey_caught)
        predators_avoided_gradient.append(predators_avoided)

    # prey_caught_cont_abl, predators_avoided = get_measures_random(model, targeted_neurons, number_of_trials)
    # print(f"Control (random ablation): Prey caught: {prey_caught_cont_abl}, Predators avoided: {predators_avoided}")
    return prey_caught_gradient, predators_avoided_gradient


get_both_measures_random_gradient("new_even_prey_ref-1", "Indiscriminate-even_prey_only", 3)
print("\n")
get_both_measures_random_gradient("new_even_prey_ref-1", "Indiscriminate-even_naturalistic", 3)

# get_both_measures("new_differential_prey_ref-4", "Prey-Sighted-differential_prey_low_predator", 3)
# get_both_measures("new_differential_prey_ref-4", "No-Response-differential_prey_low_predator", 3)

# get_both_measures("new_differential_prey_ref-4", "Prey-Sighted-differential_prey_only", 3)
# get_both_measures("new_differential_prey_ref-4", "No-Response-differential_prey_only", 3)


