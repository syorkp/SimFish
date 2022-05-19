

def extract_turn_sequences(action_sequences):
    """Takes a list of numpy arrays, these being sequences of actions. Returns only the turns contained in these."""
    valid_turns = [1, 2, 4, 5]

    new_sequences = []
    for sequence in action_sequences:
        new_sequence = [a for a in sequence if a in valid_turns]
        if len(new_sequence) > 0:
            new_sequences.append(new_sequence)

    return new_sequences