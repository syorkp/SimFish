from Analysis.load_data import load_data

data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic-1")
ro = data["observation"][:, :, 0, 0]

data2 = load_data("dqn_scaffold_18-1", "Behavioural-Data-Long-Interruptions", "Naturalistic-1")
ro = data2["observation"][:, :, 0, 0]
x = True
