
# Repeating points is removed for one step.
brief_interruption_profile = [1 if i > 100 and i % 10 == 0 else 0 for i in range(10000)]


# After an initialisation period, is removed every step

long_term_interruption_profile = [0 for i in range(100)] + [1 for i in range(10000)]


# Composed of values to input, as well as False values, for which the system updates from the previous value.
energy_state_profile = [0.5] + [False for i in range(2000)]

