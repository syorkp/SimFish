import shutil
import sys
import json


def convert_config_to_test_square(model_name, assay_config_name):
    shutil.copyfile(f"./Training-Output/{model_name}/environment_configuration.json",
                    f"./Configurations/Assay-Configs/{assay_config_name}_env.json")
    shutil.copyfile(f"./Training-Output/{model_name}/learning_configuration.json",
                    f"./Configurations/Assay-Configs/{assay_config_name}_learning.json")

    with open(f"Configurations/Assay-Configs/{assay_config_name}_env.json", 'r') as f:
        env = json.load(f)

    with open(f"Configurations/Assay-Configs/{assay_config_name}_learning.json", 'r') as f:
        params = json.load(f)

    # Change necessary parameters



    with open(f"Configurations/Assay-Configs/{assay_config_name}_env.json", 'w') as f:
        json.dump(env, f, indent=4)

    with open(f"Configurations/Assay-Configs/{assay_config_name}_learning.json", 'w') as f:
        json.dump(params, f, indent=4)

model_name = sys.argv[1]
assay_config_name = sys.argv[2]

convert_config_to_test_square(model_name, assay_config_name)





