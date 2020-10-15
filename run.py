from Legacy import training_script
from Services.training_service import TrainingService

trial = TrainingService("new_test")
trial.run()

# training_script.run("old_test")
