import shutil
import sys


def transfer_config(model_name, assay_config_name):
    shutil.copyfile(f"./Training-Output/{model_name}/environment_configuration.json",
                    f"./Configurations/Assay-Configs/{assay_config_name}_env.json")
    shutil.copyfile(f"./Training-Output/{model_name}/learning_configuration.json",
                    f"./Configurations/Assay-Configs/{assay_config_name}_learning.json")


if __name__ == "__main__":
    model_name = sys.argv[1]
    assay_config_name = sys.argv[2]

    transfer_config(model_name, assay_config_name)
