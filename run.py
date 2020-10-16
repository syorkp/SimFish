# from Legacy import training_script
import os

from Services.training_service import TrainingService

output_folder = f"./Output/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


trial = TrainingService("test2")
trial.run()

# training_script.run("old_test")
