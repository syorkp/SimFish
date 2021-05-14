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


filename = "test"


with open(f"../../Categorisation-Data/{filename}.json", 'r') as f:
    group_names = json.load(f)

display_cluster_counts(group_names)



