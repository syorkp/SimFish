

def extract_turn_sequences(action_sequences):
    """Takes a list of numpy arrays, these being sequences of actions (discrete). Returns only the turns contained in
    these."""
    valid_turns = [1, 2, 4, 5]
    percentage_retained = 0
    num_sequences = 0
    new_sequences = []
    for sequence in action_sequences:
        new_sequence = [a for a in sequence if a in valid_turns]
        if len(new_sequence) > 0:
            new_sequences.append(new_sequence)

        percentage_retained += len(new_sequence)
        num_sequences += len(sequence)
    percentage_retained = percentage_retained/num_sequences
    print(f"Percentage retained: {percentage_retained*100}")
    return new_sequences