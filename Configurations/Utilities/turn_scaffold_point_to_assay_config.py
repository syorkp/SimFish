import shutil
import sys


def transfer_config(model_name, scaffold_point, assay_config_name):
    shutil.copyfile(f"./Configurations/Training-Configs/{model_name}/{scaffold_point}_env.json",
                    f"./Configurations/Assay-Configs/{assay_config_name}_env.json")
    shutil.copyfile(f"./Configurations/Training-Configs/{model_name}/{scaffold_point}_learning.json",
                    f"./Configurations/Assay-Configs/{assay_config_name}_learning.json")


if __name__ == "__main__":
    model_name = sys.argv[1]
    scaffold_point = sys.argv[2]
    assay_config_name = sys.argv[3]

    transfer_config(model_name, scaffold_point, assay_config_name)
