
import numpy as np


def get_hunting_conditions(data, hunting_sequences, exclude_final_step=False):
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
    pre_capture_steps = np.array([i-1 for i, c in enumerate(data["consumed"]) if c == 1])

    if exclude_final_step:
        final_step = all_steps[-1]

        all_steps = np.array([i for i in all_steps if i != final_step])
        hunting_steps = np.array([i for i in hunting_steps if i != final_step])
        initiation_steps = np.array([i for i in initiation_steps if i != final_step])
        abort_steps = np.array([i for i in abort_steps if i != final_step])
        pre_capture_steps = np.array([i for i in pre_capture_steps if i != final_step])

    return all_steps, hunting_steps, initiation_steps, abort_steps, pre_capture_steps



