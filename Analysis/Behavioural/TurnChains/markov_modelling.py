import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM

from Analysis.load_data import load_data


def fitHMM(Q, nSamples, Qtest):
    """Creates a two state markov model from turn sequences as done by Dunn et al. (2016)."""

    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=3, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))

    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(Q, [len(Q), 1]))
    hidden_states_2 = model.predict(np.reshape(Qtest, [len(Qtest), 1]))

    # find parameters of Gaussian HMM
    mus = np.array(model.means_)
    sigmas = np.array(np.sqrt(np.array([np.diag(model.covars_[0]), np.diag(model.covars_[1])])))
    P = np.array(model.transmat_)

    # find log-likelihood of Gaussian HMM
    logProb = model.score(np.reshape(Q, [len(Q), 1]))

    # generate nSamples from Gaussian HMM
    samples = model.sample(nSamples)

    # re-organize mus, sigmas and P so that first row is lower mean (if not already)
    if mus[0] > mus[1]:
        mus = np.flipud(mus)
        sigmas = np.flipud(sigmas)
        P = np.fliplr(np.flipud(P))
        hidden_states = 1 - hidden_states

    return hidden_states, mus, sigmas, P, logProb, samples, hidden_states_2


if __name__ == "__main__":
    # TODO: with new, trained models, need to work out which method is best for identifying hidden states:
    # TODO:   - input angle and impulse values
    # TODO:   - input action numbers
    # TODO:   - input only turn sequences - binary lateralised

    # TODO: Also need to work out which number of states is best used to localise turn sequences.

    # TODO: Then try modifying action block plots with these states.
    # TODO: Try labelling PCA (traj) space with turn states.


    data = load_data("dqn_scaffold_14-2", "Behavioural-Data-Empty", "Naturalistic-1")
    to_use = (data["fish_position"][:, 0] > 100) * (data["fish_position"][:, 1] > 100) *\
             (data["fish_position"][:, 0] < 1400) * (data["fish_position"][:, 1] < 1400)
    actions = (data["fish_angle"][1:] - data["fish_angle"][:-1])[to_use[1:]]

    data2 = load_data("dqn_scaffold_14-2", "Behavioural-Data-Empty", "Naturalistic-2")
    to_use2 = (data2["fish_position"][:, 0] > 100) * (data2["fish_position"][:, 1] > 100) *\
              (data2["fish_position"][:, 0] < 1400) * (data2["fish_position"][:, 1] < 1400)
    actions2 = (data2["fish_angle"][1:] - data2["fish_angle"][:-1])[to_use2[1:]]

    # Convert actions to turns.
    left_turns = (actions == 1) + (actions == 4) + (actions == 7) + (actions == 10)
    right_turns = (actions == 2) + (actions == 5) + (actions == 8) + (actions == 11)
    no_turn = (actions == 0) + (actions == 3) + (actions == 6) + (actions == 9)

    lateralised_actions = np.zeros(actions.shape)
    lateralised_actions[left_turns] = 1
    # lateralised_actions[no_turn] = 0
    lateralised_actions[right_turns] = -1
    turn_sequences = lateralised_actions[lateralised_actions != 0]

    # Try running on both all sequences, then on purely exploration sequences
    all = fitHMM(actions, 100, actions2)





