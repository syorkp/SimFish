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


def display_specific_cluster_counts(groups, cluster_groups, cluster_labels):
    all_clusters = [cluster for sublist in cluster_groups for cluster in sublist]
    group_counts = [[len(groups[key][str(cluster)]) for key in groups.keys() if cluster in all_clusters] for cluster in range(len(groups[list(groups.keys())[0]].keys()))]
    narrowed_group_counts = []
    for cluster_numbers in cluster_groups:
        counts = [0 for i in range(4)]
        for cluster in cluster_numbers:
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


filename = "final_even2"
# filename = "test"


with open(f"../../Categorisation-Data/{filename}.json", 'r') as f:
    group_names = json.load(f)

placeholder_list = group_names["new_even_prey_ref-4"]["3"] + group_names["new_even_prey_ref-4"]["16"]
predator_only_ns = [2, 13, 19, 23, 27, 29, 34, 36, 46, 50, 60, 66, 81, 82, 93, 94, 95, 99, 100, 106, 110, 113, 117, 119, 122, 135, 145, 150, 156, 163, 165, 169, 171, 174, 182, 185, 186, 201, 203, 217, 218, 219, 220, 225, 226, 227, 238, 244, 259, 261, 264, 269, 280, 290, 302, 308, 310, 317, 322, 324, 339, 341, 345, 350, 366, 373, 402, 411, 450, 464, 469, 471, 477, 493]
prey_only_ns = [72, 77, 82, 138, 222, 232, 253, 268, 279, 318, 369, 382, 385, 388, 410, 433, 461, 481]
predator_cs_ns = group_names["new_even_prey_ref-4"]["15"] + group_names["new_even_prey_ref-4"]["11"]
valence_ns = group_names["new_even_prey_ref-4"]["3"] + group_names["new_even_prey_ref-4"]["9"]
prey_full_field_ns = group_names["new_even_prey_ref-4"]["7"] + group_names["new_even_prey_ref-4"]["8"] + \
                     group_names["new_even_prey_ref-4"]["9"]+ group_names["new_even_prey_ref-4"]["12"]


# display_specific_cluster_counts(group_names, [29, 25, 1, 8], [12, 13, 14, 15, 16, 28, 5], ["Prey-in-front", "Prey-Full-Field"])
display_specific_cluster_counts(group_names, [[3, 16], [5, 8, 9, 12], [3, 9], [15, 11]],
                                ["15o-in-front", "Prey-Full-Field", "Valence", "Predator-RL"])



