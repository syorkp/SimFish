import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def display_cluster_counts(groups):
    group_counts = [[len(groups[key][str(cluster)]) for key in groups.keys()] for cluster in range(len(groups[list(groups.keys())[0]].keys()))]
    data = pd.DataFrame(group_counts)
    df = pd.DataFrame(data).T
    df = df.rename(columns={k: f'Data{k + 1}' for k in range(len(data))}).reset_index()
    df = pd.wide_to_long(df, stubnames=['Data'], i='index', j='ID').reset_index()[['ID', 'Data']]
    sns.boxplot(x='ID', y='Data', data=df)
    # plt.xticks([i for i in range()], ["Prey-and-Predator", "Predator", "Prey", "Neither"])
    plt.xlabel("Cluster")
    plt.ylabel("Number of Neurons")
    plt.show()


def display_specific_cluster_counts(groups, cluster_numbers_1, cluster_numbers_2, cluster_labels):
    group_counts = [[len(groups[key][str(cluster)]) for key in groups.keys() if cluster in cluster_numbers_2 + cluster_numbers_1] for cluster in range(len(groups[list(groups.keys())[0]].keys()))]
    narrowed_group_counts = []
    counts = [0 for i in range(4)]
    for cluster in cluster_numbers_1:
        counts_a = group_counts[cluster]
        for i, count in enumerate(counts_a):
            counts[i] += count
    narrowed_group_counts.append(counts)
    counts = [0 for i in range(4)]
    for cluster in cluster_numbers_2:
        counts_a = group_counts[cluster]
        for i, count in enumerate(counts_a):
            counts[i] += count
    narrowed_group_counts.append(counts)

    sns.set()
    data = pd.DataFrame(narrowed_group_counts)
    df = pd.DataFrame(data).T
    df = df.rename(columns={k: f'Data{k + 1}' for k in range(len(data))}).reset_index()
    df = pd.wide_to_long(df, stubnames=['Data'], i='index', j='ID').reset_index()[['ID', 'Data']]
    # sns.boxplot(x='ID', y='Data', data=df)
    ax = sns.boxplot(x='ID', y='Data', data=df, fliersize=0)
    ax = sns.stripplot(y="Data", x="ID", data=df, color=".3")
    plt.xticks([i for i in range(len(cluster_labels))], cluster_labels)
    plt.xlabel("Cluster")
    plt.ylabel("Number of Neurons")
    plt.show()


filename = "test"


with open(f"../../Categorisation-Data/{filename}.json", 'r') as f:
    group_names = json.load(f)

display_specific_cluster_counts(group_names, [29, 25, 1, 8], [12, 13, 14, 15, 16, 28, 5], ["Prey-in-front", "Prey-Full-Field"])



