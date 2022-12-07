import json


original_config = "ppo_21_2"
new_config = "ppo_21_2_empty"

configuration_location = f"../Assay-Configs/{original_config}"
with open(f"{configuration_location}_learning.json", 'r') as f:
    params = json.load(f)
with open(f"{configuration_location}_env.json", 'r') as f:
    env = json.load(f)

env["prey_num"] = 0

new_configuration_location = f"../Assay-Configs/{new_config}"
with open(f"{new_configuration_location}_learning.json", 'w') as f:
    json.dump(params, f, indent=4)
with open(f"{new_configuration_location}_env.json", 'w') as f:
    json.dump(env, f, indent=4)
