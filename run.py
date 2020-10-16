import os

from Services.training_service import TrainingService

output_folder = f"./Output/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


trial = TrainingService("new_test")
trial.run()
