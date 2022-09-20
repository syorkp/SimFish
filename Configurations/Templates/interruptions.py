
# Repeating points is removed for one step.
brief_interruption_profile = [1 if i > 100 and i % 10 == 0 else 0 for i in range(10000)]


# After an initialisation period, is removed every step
long_term_visual_interruption_profile = [0 for i in range(200)] + [1 for i in range(10000)]


# Composed of values to input, as well as False values, for which the system updates from the previous value.
energy_state_profile = [0.5] + [False for i in range(2000)]
energy_state_profile_long_term = [False for i in range(200)] + [0.8 for i in range(2000)]
salt_profile_long_term = [False for i in range(200)] + [0.0 for i in range(2000)]
in_light_profile_long_term = [False for i in range(200)] + [1 for i in range(2000)]


# Reafference Interruptions
efference_A = [False for i in range(200)] + [0 for i in range(2000)]
efference_B = [False for i in range(200)] + [1 for i in range(2000)]
efference_C = [False for i in range(200)] + [2 for i in range(2000)]
efference_D = [False for i in range(200)] + [3 for i in range(2000)]
efference_E = [False for i in range(200)] + [4 for i in range(2000)]
efference_F = [False for i in range(200)] + [5 for i in range(2000)]
efference_G = [False for i in range(200)] + [6 for i in range(2000)]
efference_H = [False for i in range(200)] + [9 for i in range(2000)]

efference_V = [False for i in range(200)] + [3 for i in range(2000)]
efference_W = [False for i in range(200)] + [4 for i in range(2000)]
efference_X = [False for i in range(200)] + [5 for i in range(2000)]
efference_Y = [False for i in range(200)] + [6 for i in range(2000)]
efference_Z = [False for i in range(200)] + [9 for i in range(2000)]

# Relocating fish
fish_relocation_to_nowhere = [0 for i in range(200)] + ["E"] + [0 for i in range(10000)]


