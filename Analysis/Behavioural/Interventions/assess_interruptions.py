import numpy as np

from Analysis.load_data import load_data

from Analysis.Behavioural.VisTools.show_action_sequence_block import display_all_sequences


actions_i1 = []
actions_i2 = []
actions_i3 = []
aactions_i1 = []
aactions_i2 = []
aactions_i3 = []


for i in range(1, 4):
    data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Full-Interruptions", f"Naturalistic-{i}")
    actions_i1.append(data["action"][200:350])

    data = load_data("dqn_scaffold_14-2", "Behavioural-Data-Full-Interruptions", f"Naturalistic-{i}")
    aactions_i1.append(data["action"][200:350])

for i in range(1, 4):
    data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Full-Interruptions2", f"Naturalistic-{i}")
    actions_i2.append(data["action"][200:350])

    data = load_data("dqn_scaffold_14-2", "Behavioural-Data-Full-Interruptions2", f"Naturalistic-{i}")
    aactions_i2.append(data["action"][200:350])

for i in range(1, 4):
    data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Full-Interruptions3", f"Naturalistic-{i}")
    actions_i3.append(data["action"][200:350])

    data = load_data("dqn_scaffold_14-2", "Behavioural-Data-Full-Interruptions3", f"Naturalistic-{i}")
    aactions_i3.append(data["action"][200:350])


display_all_sequences(aactions_i1)
display_all_sequences(aactions_i2)
display_all_sequences(aactions_i3)
display_all_sequences(actions_i1)
display_all_sequences(actions_i2)
display_all_sequences(actions_i3)
