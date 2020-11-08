import os
import json
import re


from Services.training_service import TrainingService


class TrialManager:

    def __init__(self, trial_configuration):
        """
        A service that manages running of the different trial types, according to their specified environment and
        learning parameters. This is done in a way that allows threading and so simultaneous running of trials.
        Trials may be either: training, experimental, [or interactive].

        :param trial_configuration: A list, containing dictionary elements. Could use JSON if easier.
        """
        # Order the trials
        trial_configuration.sort(key=lambda item: item.get("Priority"))
        self.priority_ordered_trials = trial_configuration

        self.create_configuration_files()
        self.create_output_directories()
        self.trial_services = self.create_trial_services()
        print(self.trial_services)

        # TODO: Should also extract all information that needs to be maintained across trials to allow purity of
        #  training service and prevent code repeats.
        # TODO: In fact, should do everything that is repeated over the different services not related to the learning
        #  algorithm.

    def create_configuration_files(self):
        """
        For each of the trials specified, creates the required environment and learning configuration JSON files by
        running the existing create_configuration_[].py files.
        :return:
        """
        for trial in self.priority_ordered_trials:
            configuration_creator_file = f"Configurations/create_configuration_{trial['Environment Name']}.py"
            os.system(configuration_creator_file)

    def create_output_directories(self):
        """
        If there are not already existing output directories for the trials, creates them.
        :return:
        """
        print("Checking whether any of the trial models exist...")
        for index, trial in enumerate(self.priority_ordered_trials):
            output_directory_location = f"./Output/{trial['Environment Name']}_{trial['Trial Number']}_output"
            if not os.path.exists(output_directory_location):
                os.makedirs(output_directory_location)
                os.makedirs(f"{output_directory_location}/episodes")
                os.makedirs(f"{output_directory_location}/logs")
                self.priority_ordered_trials[index]["Model Exists"] = False
            elif self.check_model_exists(output_directory_location):
                self.priority_ordered_trials[index]["Model Exists"] = True
            else:
                self.priority_ordered_trials[index]["Model Exists"] = False
        print(self.priority_ordered_trials)

    @staticmethod
    def  check_model_exists(output_directory_location):
        """Checks if a model checkpoint has been saved."""
        output_file_contents = os.listdir(output_directory_location)
        for name in output_file_contents:
            if ".cptk.index" in name:
                return True
        return False

    @staticmethod
    def load_configuration_files(environment_name):
        """
        Called by create_trials method, should return the learning and environment configurations in JSON format.
        :param environment_name:
        :return:
        """
        print("Loading configuration...")
        configuration_location = f"./Configurations/JSON-Data/{environment_name}"
        with open(f"{configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    def create_trial_services(self):
        """
        Creates the instances of TrainingService and ExperimentService required for the trial configuration.
        :return:
        """
        # Pass in to services: Whether or not a model already exists, any other carried on information
        # e.g. episode number, epsilon value.
        trial_services = []
        for trial in self.priority_ordered_trials:
            learning_params, environment_params = self.load_configuration_files(trial["Environment Name"])
            if trial["Model Exists"]:
                output_directory_location = f"./Output/{trial['Environment Name']}_{trial['Trial Number']}_output"
                with open(f"{output_directory_location}/saved_parameters.json", "r") as file:
                    data = json.load(file)
                    # Could also just pass in data and assign within the service.
                    epsilon = data["epsilon"]
                    total_steps = data["total_steps"]
                    episode_number = data["episode_number"]
            else:
                epsilon = None
                total_steps = None
                episode_number = None
            # TODO: Standardise the way this is done and add to bespoke function.
            if trial["Run Mode"] == "Training":
                trial_services.append(TrainingService(environment_name=trial["Environment Name"],
                                                      trial_number=trial["Trial Number"],
                                                      model_exists=trial["Model Exists"],
                                                      fish_mode=trial["Fish Setup"],
                                                      learning_params=learning_params,
                                                      env_params=environment_params,
                                                      e=epsilon,
                                                      total_steps=total_steps,
                                                      episode_number=episode_number,
                                                      )
                                      )
            elif trial["Run Mode"] == "Experimental":
                pass  # TODO: Add in experiment_service here.
        return trial_services

    def run_priority_loop(self):
        """
        Executes the trials in the required order.
        :return:
        """
        # TODO: Add in threading and parallelism.
        # TODO: Check use of GPU and whether another thread should be dropped.
        for service in self.trial_services:
            service.run()







