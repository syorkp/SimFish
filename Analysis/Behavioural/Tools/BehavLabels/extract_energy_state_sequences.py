

def label_energy_state_within_range(data, e_min, e_max):
    energy_state = data["energy_state"]
    below_max = energy_state < e_max
    above_min = energy_state >= e_min
    within_range = (below_max * above_min) * 1
    return within_range