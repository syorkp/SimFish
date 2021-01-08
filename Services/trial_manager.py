import os
import json
import multiprocessing

import Services.training_service as training
import Services.assay_service as assay


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

        # self.create_configuration_files()  TODO: Possibly add later if makes sense to.
        self.create_output_directories()

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
            output_directory_location = f"./Training-Output/{trial['Model Name']}-{trial['Trial Number']}"
            assay_directory_location = f"./Assay-Output/{trial['Model Name']}-{trial['Trial Number']}"

            if trial["Run Mode"] == "Training":
                if not os.path.exists(output_directory_location):
                    os.makedirs(output_directory_location)
                    os.makedirs(f"{output_directory_location}/episodes")
                    os.makedirs(f"{output_directory_location}/logs")
                    self.priority_ordered_trials[index]["Model Exists"] = False
                elif self.check_model_exists(output_directory_location):
                    self.priority_ordered_trials[index]["Model Exists"] = True
                else:
                    self.priority_ordered_trials[index]["Model Exists"] = False
            elif trial["Run Mode"] == "Assay":
                self.priority_ordered_trials[index]["Model Exists"] = True
                if not os.path.exists(assay_directory_location):
                    os.makedirs(assay_directory_location)
        print(self.priority_ordered_trials)

    @staticmethod
    def check_model_exists(output_directory_location):
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
        configuration_location = f"./Configurations/Assay-Configs/{environment_name}"
        with open(f"{configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    @staticmethod
    def get_saved_parameters(trial):
        """
        Extracts the saved parameters in teh saved_parameters.json document.
        :return:
        """
        if trial["Model Exists"]:
            output_directory_location = f"./Training-Output/{trial['Model Name']}-{trial['Trial Number']}"
            with open(f"{output_directory_location}/saved_parameters.json", "r") as file:
                data = json.load(file)
                epsilon = data["epsilon"]
                total_steps = data["total_steps"]
                episode_number = data["episode_number"]
        else:
            epsilon = None
            total_steps = None
            episode_number = None

        return epsilon, total_steps, episode_number

    def run_priority_loop(self):
        """
        Executes the trials in the required order.
        :return:
        """
        parallel_jobs = 2
        memory_fraction = 0.99/parallel_jobs
        running_jobs = {}
        for index, trial in enumerate(self.priority_ordered_trials):
            epsilon, total_steps, episode_number = self.get_saved_parameters(trial)
            if trial["Run Mode"] == "Training":
                running_jobs[str(index)] = multiprocessing.Process(target=training.training_target, args=(trial, epsilon, total_steps, episode_number, memory_fraction))
            elif trial["Run Mode"] == "Assay":
                learning_params, environment_params = self.load_configuration_files(trial["Environment Name"])
                running_jobs[str(index)] = multiprocessing.Process(target=assay.assay_target, args=(trial, learning_params, environment_params, total_steps, episode_number, memory_fraction))
            running_jobs[str(index)].start()
            print(f"Jobs: {running_jobs}")
            while len(running_jobs.keys()) > parallel_jobs - 1:
                for process in running_jobs.keys():
                    if running_jobs[process].is_alive():
                        pass
                    else:
                        running_jobs[str(index)].join()
                        del running_jobs[process]


