from Analysis.Behavioural.New.display_action_sequences import display_all_sequences_capture, get_capture_sequences


def extract_attempted_captures():
    ...

cs = get_capture_sequences("new_even_prey_ref-4", "Ablation-Test-Predator_Only-behavioural_data", "Random-Control", 12)
display_all_sequences_capture(cs[:26])
x = True
