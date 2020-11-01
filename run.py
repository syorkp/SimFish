import os

from Services.training_service import TrainingService


# Specifications for simulation
environment_name = "base"
trial_number = "1"

output_folder = "./Output/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Run the configuration creator
configuration_creator_file = f"Configurations/create_configuration_{environment_name}.py"

os.system(configuration_creator_file)


trial = TrainingService()
trial.run()
