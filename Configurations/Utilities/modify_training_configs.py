import json


def modify_training_configs(config_name, variable, new_value, num_configs, prefix="env"):
    """Changes a variable throughout all env config files for a training scaffold."""
    config_location = f"../Training-Configs/{config_name}"

    for config in range(1, num_configs + 1):
        with open(f"{config_location}/{config}_{prefix}.json", "r") as file:
            data = json.load(file)
        data[variable] = new_value
        with open(f"{config_location}/{config}_{prefix}.json", "w") as outfile:
            json.dump(data, outfile, indent=4)


if __name__ == "__main__":
    modify_training_configs("ppo_gamma", "predator_impulse", 5, 60, prefix="env")
    modify_training_configs("dqn_gamma", "predator_impulse", 5, 52, prefix="env")
    # modify_training_configs("dqn_scaffold_30", "num_episodes", 100000, 43, prefix="learning")
    # modify_training_configs("dqn_scaffold_30_even_p", "num_episodes", 100000, 43, prefix="learning")
    # modify_training_configs("dqn_scaffold_30_fixed_p", "num_episodes", 100000, 43, prefix="learning")
    # modify_training_configs("dqn_scaffold_30_noiseless", "num_episodes", 100000, 42, prefix="learning")
    # modify_training_configs("dqn_scaffold_30_static_p", "num_episodes", 100000, 43, prefix="learning")
    # modify_training_configs("dqn_scaffold_33", "num_episodes", 100000, 47, prefix="learning")
    # modify_training_configs("dqn_scaffold_34", "num_episodes", 100000, 47, prefix="learning")
    # modify_training_configs("ppo_scaffold_21", "num_episodes", 100000, 96, prefix="learning")

    # modify_training_configs("dqn_scaffold_26", "ci", 4e-07, 43, prefix="env")
    # modify_training_configs("dqn_scaffold_30", "ci", 4e-07, 43, prefix="env")
    #
    # modify_training_configs("dqn_scaffold_26", "ca", 4e-07, 43, prefix="env")
    # modify_training_configs("dqn_scaffold_30", "ca", 4e-07, 43, prefix="env")
    #
    # modify_training_configs("dqn_scaffold_26", "startE", 0.3, 43, prefix="learning")
    # modify_training_configs("dqn_scaffold_30", "startE", 0.3, 43, prefix="learning")

