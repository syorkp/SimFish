
import numpy as np


def get_hunting_conditions(data, hunting_sequences):
    failed_sequences = []
    for ts in hunting_sequences:
        if data["consumed"][ts[-1]+1]:
            pass
        else:
            failed_sequences.append(ts)

    all_steps = np.array([i for i in range(len(data["consumed"]))])
    hunting_steps = np.array(list(set([a for at in hunting_sequences for a in at])))
    initiation_steps = np.array(list(set([a[0] for a in hunting_sequences])))
    abort_steps = np.array(list(set([a[-1] for a in hunting_sequences])))

    return all_steps, hunting_steps, initiation_steps, abort_steps



