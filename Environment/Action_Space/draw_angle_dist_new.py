import numpy as np


def draw_angle_dist_new(bout_id):
    if bout_id == 8:  # Slow2
        mean = [2.49320953, 0.0910835]
        cov = [[4.24887401e-01, 3.07099392e-04],
               [3.07099392e-04, 1.45687655e-03]]
    elif bout_id == 7:  # RT
        mean = [2.74619216, 0.82713249]
        cov = [[0.3839484,  0.02302918],
              [0.02302918, 0.03937928]]
    elif bout_id == 0:  # sCS
        mean = [0.95660315, 0.11893887]
        cov = [[0.07604165, 0.00540778],
               [0.00540778, 0.00560931]]
    elif bout_id == 4:  # J-turn 1
        mean = [0.49074911, 0.39750791]
        cov = [[0.00679925, 0.00071446],
               [0.00071446, 0.00626601]]
    elif bout_id == 44:  # J-turn 2
        mean = [1.0535197,  0.61945679]
        cov = [[0.0404599, -0.00318193],
               [-0.00318193, 0.01365224]]
    elif bout_id == 5:  # C-Start
        mean = [7.03322223, 0.67517832]
        cov = [[1.35791922, 0.10690938],
               [0.10690938, 0.10053853]]
    elif bout_id == 10:  # AS
        mean = [0.64204809, 0.07020727]
        cov = [[0.04002195, -0.00028982],
               [-0.00028982, 0.00160061]]
    else:
        mean = [0, 0],
        cov = [[0, 0],
              [0, 0]]
        print("Draw action error")

    bout_vals = np.random.multivariate_normal(mean, cov, 1)
    return bout_vals[0, 0], bout_vals[0, 1]


if __name__ == "__main__":
    x  = draw_angle_dist_new(10)
    x  = draw_angle_dist_new(5)


