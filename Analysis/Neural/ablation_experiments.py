from Analysis.load_data import load_data


data = load_data("even_prey_ref-5", "Ablation-Test-1", "Naturalistic-1")
# data2 = load_data("even_prey_ref-5", "Ablation-Test-2", "Naturalistic-1")
# data3 = load_data("even_prey_ref-5", "Ablation-Test-3", "Naturalistic-1")
# data4 = load_data("even_prey_ref-5", "Ablation-Test-4", "Naturalistic-1")
data2 = load_data("even_prey_ref-5", "Ablation-Test-Prey-Only", "Naturalistic-1")
data3 = load_data("even_prey_ref-5", "Ablation-Test-Pred-Only", "Naturalistic-1")
data4 = load_data("even_prey_ref-5", "Ablation-Test-Prey-Only-Random", "Naturalistic-1")
data6 = load_data("even_prey_ref-5", "Ablation-Test-Unvexed", "Naturalistic-1")
data7 = load_data("even_prey_ref-5", "Ablation-Test-Unvexed-Random", "Naturalistic-1")
print(f"Vexed ablation total = {sum(data6['consumed']), }  "
       f"Random total = {sum(data7['consumed'])}, ")

data5 = load_data("even_prey_ref-5", "Ablation-Test-Pred-Only-Random", "Naturalistic-1")
print(f"No ablation total = {sum(data['consumed']), }  "
      f"Prey only Ablation total = {sum(data2['consumed'])}, "
      f"Pred only Random ablation total = {sum(data3['consumed'])}, "
      f"Prey only random ablation total = {sum(data4['consumed'])}, "
      f"Pred only random ablation  total = {sum(data5['consumed'])}")
