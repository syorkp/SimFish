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


def get_measures(model, targeted_neurons, percentage, number_of_trials):
    prey_caught = 0
    predators_avoided = 0
    for i in range(1, number_of_trials+1):
        data1 = load_data(model, f"Ablation-Test-{targeted_neurons}", f"Prey-Only-Ablated-{percentage}-{i}")
        prey_caught = prey_caught + sum(data1['consumed'])
        predators_avoided = predators_avoided + get_predator_num(data1)
    return prey_caught/number_of_trials, predators_avoided/number_of_trials


def get_measures_2(model, targeted_neurons, number_of_trials):
    prey_caught = 0
    predators_avoided = 0
    for i in range(1, number_of_trials+1):
        data1 = load_data(model, f"Ablation-Test-{targeted_neurons}", f"Prey-Only-{i}")
        prey_caught = prey_caught + sum(data1['consumed'])
        predators_avoided = predators_avoided + get_predator_num(data1)
    return prey_caught/number_of_trials, predators_avoided


def get_measures_3(model, targeted_neurons, number_of_trials):
    prey_caught = 0
    predators_avoided = 0
    for i in range(1, number_of_trials+1):
        data1 = load_data(model, f"Ablation-Test-{targeted_neurons}", f"Prey-Only-Control-{i}")
        prey_caught = prey_caught + sum(data1['consumed'])
        predators_avoided = predators_avoided + get_predator_num(data1)
    return prey_caught/number_of_trials, predators_avoided


def get_both_measures(model, targeted_neurons, number_of_trials):
    gradient = []
    for per in range(10, 110, 10):
        prey_caught, predators_avoided = get_measures(model, targeted_neurons, per, number_of_trials)
        print(f"Ablations: {per}, Prey caught: {prey_caught}, Predators avoided: {predators_avoided}")
        gradient.append(prey_caught)

    prey_caught_no_abl, predators_avoided = get_measures_2(model, targeted_neurons, number_of_trials)
    print(f"Control (no ablation): Prey caught: {prey_caught_no_abl}, Predators avoided: {predators_avoided}")
    prey_caught_cont_abl, predators_avoided = get_measures_3(model, targeted_neurons, number_of_trials)
    print(f"Control (random ablation): Prey caught: {prey_caught_cont_abl}, Predators avoided: {predators_avoided}")
    return gradient, prey_caught_cont_abl, prey_caught_no_abl

# get_both_measures("even_prey_ref-7", "Predator-Only", 3)




