
"""Scripts to demonstrate efficiency for exploration of markov model parameters.
- Given parameters of markov model, generates example sequences.
- Then generates example sequences for deviations of markov model transition probabilities.
- Sequences are run in stripped down environment and an exploration quotient is calculated for each.

Problem: Would result in wall collisions that could be avoided in a model. Could either make an environment without walls,
or simply reflect fish from walls.

"""


