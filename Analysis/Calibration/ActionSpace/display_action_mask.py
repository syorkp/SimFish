import numpy as np
import matplotlib.pyplot as plt

from Environment.Action_Space.Bout_classification.action_masking import get_action_mask


if __name__ == "__main__":
    kde_i, kde_a, accepted_bouts = get_action_mask()

    x = np.linspace(0, 100, num=400)
    y = np.linspace(0, 5, num=400)
    X, Y = np.meshgrid(x, y)
    X_flat = X.flatten()
    Y_flat = Y.flatten()

    i_probs = kde_i.pdf(X_flat).reshape((400, 400))
    a_probs = kde_a.pdf(Y_flat).reshape((400, 400))
    probs = i_probs * a_probs
    # accepted_region = (probs >= 0.0000389489489) * 1.0
    probs **= 0.1
    accepted_region = probs
    accepted_region = accepted_region[::-1, :]

    plt.imshow(accepted_region, extent=[0, 100, 0, 5])
    # plt.scatter(accepted_bouts[:, 0], accepted_bouts[:, 1])
    plt.show()


