import numpy as np


def draw_angle_dist_new(bout_id):
    if bout_id == 8:  # Slow2
        mean = [2.49320953e+00, 2.36217665e-19]
        cov = [[4.24434912e-01, 1.89175382e-18],
                [1.89175382e-18, 4.22367139e-03]]
    elif bout_id == 7:  # RT
        mean = [2.74619216, 0.82713249]
        cov = [[0.3839484,  0.02302918],
               [0.02302918, 0.03937928]]
    elif bout_id == 0:  # sCS
        mean = [0.956603146, -6.86735892e-18]
        cov = [[2.27928786e-02, 1.52739195e-19],
               [1.52739195e-19, 3.09720798e-03]]
    elif bout_id == 4:  # J-turn 1
        mean = [0.49074911, 0.39750791]
        cov = [[0.00679925, 0.00071446],
               [0.00071446, 0.00626601]]
    elif bout_id == 44:  # J-turn 2
        mean = [1.0535197,  0.61945679]
        # cov = [[ 0.0404599,  -0.00318193],
        #        [-0.00318193,  0.01365224]]
        cov = [[0.0404599,  0.0],
               [0.0,  0.01365224]]
    elif bout_id == 5:  # C-Start
        mean = [7.03322223, 0.67517832]
        cov = [[1.35791922, 0.10690938],
               [0.10690938, 0.10053853]]
    elif bout_id == 10:  # AS
        mean = [6.42048088e-01, 1.66490488e-17]
        cov = [[3.99909515e-02, 3.58321400e-19],
               [3.58321400e-19, 3.24366068e-03]]
    else:
        mean = [0, 0],
        cov = [[0, 0],
               [0, 0]]
        print("Draw action error")

    bout_vals = np.random.multivariate_normal(mean, cov, 1)
    return bout_vals[0, 1], bout_vals[0, 0]


if __name__ == "__main__":
    x = draw_angle_dist_new(10)
    x = draw_angle_dist_new(5)


