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
    modify_training_configs("dqn_scaffold_33", "sand_grain_num", 50, 47, prefix="env")
    modify_training_configs("dqn_scaffold_34", "wall_touch_penalty", 50, 47, prefix="env")

    # modify_training_configs("dqn_scaffold_26", "ci", 4e-07, 43, prefix="env")
    # modify_training_configs("dqn_scaffold_30", "ci", 4e-07, 43, prefix="env")
    #
    # modify_training_configs("dqn_scaffold_26", "ca", 4e-07, 43, prefix="env")
    # modify_training_configs("dqn_scaffold_30", "ca", 4e-07, 43, prefix="env")
    #
    # modify_training_configs("dqn_scaffold_26", "startE", 0.3, 43, prefix="learning")
    # modify_training_configs("dqn_scaffold_30", "startE", 0.3, 43, prefix="learning")

