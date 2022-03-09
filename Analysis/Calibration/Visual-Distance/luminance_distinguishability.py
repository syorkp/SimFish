import matplotlib.pyplot as plt
import numpy as np


with open(f'Luminance-Distinguishability/distinguishability_scores_full.npy', 'rb') as outfile:
    distinguishability_scores_full = np.load(outfile)

with open(f'Luminance-Distinguishability/distinguishability_scores_partial.npy', 'rb') as outfile:
    distinguishability_scores_partial = np.load(outfile)

with open(f'Luminance-Distinguishability/uv_stimulus_photons_full.npy', 'rb') as outfile:
    uv_stimulus_photons_full = np.load(outfile)

with open(f'Luminance-Distinguishability/uv_stimulus_photons_partial.npy', 'rb') as outfile:
    uv_stimulus_photons_partial = np.load(outfile)


def plot_only_positive(luminance_values, distinguishability_scores_full, distinguishability_scores_partial):
    luminance_full = []
    luminance_partial = []
    sorted_distinguishability_scores_full = []
    sorted_distinguishability_scores_partial = []
    for i, l in enumerate(luminance_values):
        if l < 0.265:
            pass
        else:
            if i == len(luminance_values) -1:
                break
            if distinguishability_scores_full[i] < distinguishability_scores_full[i+1]:
                luminance_full.append(l)
                sorted_distinguishability_scores_full.append(distinguishability_scores_full[i])
            if distinguishability_scores_partial[i] < distinguishability_scores_partial[i+1]:
                luminance_partial.append(l)
                sorted_distinguishability_scores_partial.append(distinguishability_scores_partial[i])

    plt.plot(luminance_full, sorted_distinguishability_scores_full)
    plt.plot(luminance_partial, sorted_distinguishability_scores_partial)
    plt.show()

    # Find differences betwen two lines:
    differences = []
    luminances = []
    for l in luminance_values:
        if l in luminance_full and l in luminance_partial:
            lf_n = luminance_full.index(l)
            lp_n = luminance_partial.index(l)
            difference = abs(sorted_distinguishability_scores_partial[lp_n] - sorted_distinguishability_scores_full[lf_n])
            luminances.append(l)
            differences.append(difference)

    plt.plot(luminances, differences)
    plt.show()

    index_diff = 0
    diff = 1
    for i in range(len(differences)):
        if abs(differences[i]-0.25) < diff:
            diff = differences[i]-0.25
            index_diff = i
    print(index_diff)
    print(luminances[index_diff])

    plt.plot(luminance_full, sorted_distinguishability_scores_full)
    plt.plot(luminance_partial, sorted_distinguishability_scores_partial)
    plt.ylabel("p prey stimulus present")
    plt.xlabel("Luminance")
    plt.legend([f"3.4mm", f"10.0mm", "Normal Luminance"])
    plt.vlines(luminances[index_diff], ymin=min(sorted_distinguishability_scores_full), ymax=max(sorted_distinguishability_scores_full), color="g")
    plt.show()

    x = True

    plt.plot(luminance_full, sorted_distinguishability_scores_full)
    plt.plot(luminance_partial, sorted_distinguishability_scores_partial)
    plt.ylabel("p prey stimulus present")
    plt.xlabel("Luminance")
    plt.legend([f"3.4mm", f"10.0mm", "Normal Luminance"])
    plt.vlines(luminance_full[10], ymin=min(sorted_distinguishability_scores_full), ymax=max(sorted_distinguishability_scores_full), color="g")
    plt.show()
    print(luminance_full[10])


min_luminance = 0.25
max_luminance = 0.28
luminance_values = np.linspace(min_luminance, max_luminance, len(distinguishability_scores_full))
plot_only_positive(luminance_values, distinguishability_scores_full, distinguishability_scores_partial)
