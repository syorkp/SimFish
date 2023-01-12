import numpy as np
import matplotlib.pyplot as plt

from hmmlearn.hmm import GaussianHMM

from Analysis.load_data import load_data


def fitHMM(Q, nSamples):
    # fit Gaussian HMM to Q
    model = GaussianHMM(n_components=2, n_iter=1000).fit(np.reshape(Q, [len(Q), 1]))

    # classify each observation as state 0 or 1
    hidden_states = model.predict(np.reshape(Q, [len(Q), 1]))

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

    return hidden_states, mus, sigmas, P, logProb, samples


if __name__ == "__main__":
    data = load_data("dqn_gamma-1", "Behavioural-Data-Free", "Naturalistic-1")

    # Convert actions to turns.

    all = fitHMM(data["action"], 100)





