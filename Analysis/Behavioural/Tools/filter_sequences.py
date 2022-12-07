def remove_sCS_heavy(sequences, max_sCS=7):
    new_sequences = []
    for sequence in sequences:
        if list(sequence).count(3) > max_sCS:
            pass
        else:
            new_sequences.append(sequence)
    return new_sequences
