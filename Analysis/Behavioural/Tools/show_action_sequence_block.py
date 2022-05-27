import matplotlib.pyplot as plt
import seaborn as sns


def display_sequences(sequences):
    plot_dim = len(sequences[0])
    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'y', "k", "m", "m", "k"]
    plt.figure(figsize=(5, 15))
    for i, seq in enumerate(reversed(sequences)):
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            plt.fill_between((j, j + 1), i, i + 1, color=color_set[a])
    plt.axis("scaled")
    plt.show()


def display_all_sequences(sequences, min_length=None, max_length=None):
    sns.set()
    sequences.sort(key=len)
    plot_dim = max([len(seq) for seq in sequences])
    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'y', "k", "m", "m", "b"]
    plt.figure(figsize=(5, 15))
    for i, seq in enumerate(sequences):
        if min_length is not None:
            if len(seq) < min_length:
                continue
        if max_length is not None:
            if len(seq) > max_length:
                continue
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            plt.fill_between((j, j+1), i, i+1, color=color_set[a])
    plt.axis("scaled")
    plt.show()
