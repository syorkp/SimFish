import os
import json
import shutil
import multiprocessing

import Services.DQN.dqn_training_service as training
import Services.DQN.dqn_assay_service as assay

import Services.PPO.ppo_training_service_continuous as ppo_training_continuous
import Services.PPO.ppo_assay_service_continuous as ppo_assay_continuous
import Services.PPO.ppo_training_service_continuous_sbe as ppo_training_continuous_2
import Services.PPO.ppo_training_service_discrete as ppo_training_discrete
import Services.PPO.ppo_assay_service_discrete as ppo_assay_discrete
import Services.PPO.ppo_training_service_discrete_2 as ppo_training_discrete2

from Analysis.Training.get_checkpoints_from_scaffold_points import get_checkpoint
from Configurations.Utilities.turn_scaffold_point_to_assay_config import transfer_config

# multiprocessing.set_start_method('spawn', force=True)


class TrialManager:

    def __init__(self, trial_configuration, parallel_jobs):
        """
        A service that manages running of the different trial types, according to their specified environment and
        learning parameters. This is done in a way that allows threading and so simultaneous running of trials.
        Trials may be either: training, experimental, [or interactive].

        :param trial_configuration: A list, containing dictionary elements. Could use JSON if easier.
        """
        # Order the trials (if specified
        priority_specified = True
        for trial in trial_configuration:
            if "Priority" not in trial:
                priority_specified = False
                break

        if priority_specified:
            trial_configuration.sort(key=lambda item: item.get("Priority"))

        self.priority_ordered_trials = trial_configuration

        self.create_output_directories()

        self.parallel_jobs = parallel_jobs

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
            elif trial["Run Mode"] == "Assay-Analysis-Across-Scaffold":
                self.priority_ordered_trials[index]["Model Exists"] = True
                if not os.path.exists(assay_directory_location):
                    os.makedirs(assay_directory_location)
            elif trial["Run Mode"] == "Split-Assay":
                self.priority_ordered_trials[index]["Model Exists"] = True
                if not os.path.exists(assay_directory_location):
                    os.makedirs(assay_directory_location)
        # print(self.priority_ordered_trials)

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
        try:
            output_directory_location = f"./Training-Output/{trial['Model Name']}-{trial['Trial Number']}"
            with open(f"{output_directory_location}/saved_parameters.json", "r") as file:
                data = json.load(file)
                if "Continuous Actions" in trial:
                    if trial["Continuous Actions"]:
                        epsilon = None
                    else:
                        epsilon = data["epsilon"]
                else:
                    epsilon = None
                total_steps = data["total_steps"]
                episode_number = data["episode_number"]
                configuration_index = data["configuration_index"]
        except FileNotFoundError:
            epsilon = None
            total_steps = None
            episode_number = None
            configuration_index = None

        return epsilon, total_steps, episode_number, configuration_index

    def get_new_job(self, trial, total_steps, episode_number, memory_fraction, epsilon, configuration_index):
        # Setting optional variables
        if "Continuous Actions" in trial:
            continuous_actions = trial["Continuous Actions"]
        else:
            if trial["Learning Algorithm"] == "PPO":
                continuous_actions = True
            else:
                continuous_actions = False

        if "SB Emulator" in trial:
            sb_emulator = trial["SB Emulator"]
        else:
            sb_emulator = True

        if trial["Run Mode"] == "Training":
            if continuous_actions:
                if trial["Learning Algorithm"] == "PPO":
                    if sb_emulator:
                        new_job = multiprocessing.Process(target=ppo_training_continuous_2.ppo_training_target_continuous_sbe,
                                                          args=(trial, total_steps, episode_number, memory_fraction, configuration_index))
                    else:
                        new_job = multiprocessing.Process(target=ppo_training_continuous.ppo_training_target_continuous,
                                                      args=(trial, total_steps, episode_number, memory_fraction, configuration_index))
                elif trial["Learning Algorithm"] == "DQN":
                    print('Cannot use DQN with continuous actions (training mode)')
                    new_job = None
                else:
                    print('Invalid "Learning Algorithm" selected with continuous actions (training mode)')
                    new_job = None

            else:
                if trial["Learning Algorithm"] == "PPO":
                    if sb_emulator:
                        new_job = multiprocessing.Process(target=ppo_training_discrete2.ppo_training_target_discrete,
                                                          args=(trial, total_steps, episode_number, memory_fraction,
                                                                configuration_index))
                    else:
                        new_job = multiprocessing.Process(target=ppo_training_discrete.ppo_training_target_discrete,
                                                          args=(trial, total_steps, episode_number, memory_fraction, configuration_index))
                elif trial["Learning Algorithm"] == "DQN":
                    new_job = multiprocessing.Process(target=training.training_target,
                                                      args=(trial, epsilon, total_steps, episode_number, memory_fraction, configuration_index))
                else:
                    print('Invalid "Learning Algorithm" selected with discrete actions (training mode)')
                    new_job = None

        elif trial["Run Mode"] == "Assay-Analysis-Across-Scaffold":

            if continuous_actions:
                if trial["Learning Algorithm"] == "PPO":
                    new_job = multiprocessing.Process(target=ppo_assay_continuous.ppo_assay_target_continuous, args=(
                        trial, total_steps, episode_number, memory_fraction))

                elif trial["Learning Algorithm"] == "DQN":
                    print('Cannot use DQN with continuous actions (assay mode)')
                    new_job = None
                else:
                    print('Invalid "Learning Algorithm" selected with continuous actions (assay mode)')
                    new_job = None

            else:
                if trial["Learning Algorithm"] == "PPO":
                    new_job = multiprocessing.Process(target=ppo_assay_discrete.ppo_assay_target_discrete, args=(
                        trial, total_steps, episode_number, memory_fraction))
                elif trial["Learning Algorithm"] == "A2C":
                    print('Cannot use A2C with discrete actions (assay mode)')
                    new_job = None
                elif trial["Learning Algorithm"] == "DQN":
                    new_job = multiprocessing.Process(target=assay.assay_target, args=(
                        trial, total_steps, episode_number, memory_fraction))
                else:
                    print('Invalid "Learning Algorithm" selected with discrete actions (assay mode)')
                    new_job = None

        elif trial["Run Mode"] == "Split-Assay":
            if continuous_actions:
                if trial["Learning Algorithm"] == "PPO":
                    new_job = multiprocessing.Process(target=ppo_assay_continuous.ppo_assay_target_continuous, args=(
                        trial, total_steps, episode_number, memory_fraction))
                elif trial["Learning Algorithm"] == "DQN":
                    print('Cannot use DQN with continuous actions (assay mode)')
                    new_job = None
                else:
                    print('Invalid "Learning Algorithm" selected with continuous actions (assay mode)')
                    new_job = None

            else:
                if trial["Learning Algorithm"] == "PPO":
                    new_job = multiprocessing.Process(target=ppo_assay_discrete.ppo_assay_target_discrete, args=(
                        trial, total_steps, episode_number, memory_fraction))
                elif trial["Learning Algorithm"] == "DQN":
                    new_job = multiprocessing.Process(target=assay.assay_target, args=(
                        trial, total_steps, episode_number, memory_fraction))
                else:
                    print('Invalid "Learning Algorithm" selected with discrete actions (assay mode)')
                    new_job = None

        elif trial["Run Mode"] == "Assay":
            if continuous_actions:
                if trial["Learning Algorithm"] == "PPO":
                    new_job = multiprocessing.Process(target=ppo_assay_continuous.ppo_assay_target_continuous, args=(
                        trial, total_steps, episode_number, memory_fraction))
                elif trial["Learning Algorithm"] == "DQN":
                    print('Cannot use DQN with continuous actions (assay mode)')
                    new_job = None
                else:
                    print('Invalid "Learning Algorithm" selected with continuous actions (assay mode)')
                    new_job = None

            else:
                if trial["Learning Algorithm"] == "PPO":
                    new_job = multiprocessing.Process(target=ppo_assay_discrete.ppo_assay_target_discrete, args=(
                        trial, total_steps, episode_number, memory_fraction))
                elif trial["Learning Algorithm"] == "DQN":
                    new_job = multiprocessing.Process(target=assay.assay_target, args=(
                        trial, total_steps, episode_number, memory_fraction))
                else:
                    print('Invalid "Learning Algorithm" selected with discrete actions (assay mode)')
                    new_job = None
        else:
            print('Invalid "Run Mode" selected')
            new_job = None

        return new_job

    def run_analysis_across_scaffold(self, index, trial, running_jobs, memory_fraction, to_delete):
        complete = False

        # Find number of scaffold points
        epsilon, total_steps, episode_number, configuration = self.get_saved_parameters(trial)
        model_name = f"{trial['Model Name']}-{trial['Trial Number']}"

        # Get list of all checkpoints
        all_checkpoints = [get_checkpoint(model_name=model_name,
                                          scaffold_point=s) for s in range(2, configuration)]

        new_data_files = []

        # Iterate
        for i, chkpt in enumerate(all_checkpoints):
            complete = False
            configuration = i + 1
            episode_number = chkpt
            assay_config_name = f"{trial['Model Name']}_c{configuration}"

            # Create assay config for that scaffold point
            transfer_config(model_name=trial["Model Name"],
                            scaffold_point=configuration,
                            assay_config_name=assay_config_name)

            # If specified, modify the config
            if trial["Config Modification"] is not None and trial["Config Modification"] is not False:
                print("Altering assay config...")
                with open(f"Configurations/Assay-Configs/{assay_config_name}_env.json", "r") as f:
                    env = json.load(f)
                if trial["Config Modification"] == "Empty":
                    env["prey_num"] = 0
                    env["max_salt_damage"] = 0.
                    env["max_current_strength"] = 0.0
                    env["probability_of_predator"] = 0.
                    env["dark_light_ratio"] = 0.0
                with open(f"Configurations/Assay-Configs/{assay_config_name}_env.json", 'w') as f:
                    json.dump(env, f)

            # Change parameters of trial to match the given checkpoint
            trial["Environment Name"] = assay_config_name
            trial["Assay Configuration Name"] = assay_config_name
            trial["Checkpoint"] = chkpt

            # Log a file to be deleted
            new_data_files.append(f"Assay-Output/{model_name}/{assay_config_name}.h5")

            # 1 - ASSAY: Create assay data for that trial
            new_job = self.get_new_job(trial, total_steps, episode_number, memory_fraction, epsilon,
                                       configuration)
            if new_job is not None:
                running_jobs[str(index)] = new_job
                running_jobs[str(index)].start()
                print(f"Starting {trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']},"
                      f" Scaffold Point {configuration}")
            else:
                print("New job failed")

            while len(running_jobs.keys()) > self.parallel_jobs - 1 and to_delete is None:
                for process in running_jobs.keys():
                    if running_jobs[process].is_alive():
                        pass
                    else:
                        to_delete = process
                        running_jobs[str(index)].join()
                        print(f"{trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']} Complete")
                        complete = True

            # Clear up jobs
            if to_delete is not None:
                del running_jobs[to_delete]
                to_delete = None

            #  2 - ANALYSIS
            if complete:
                # Do analysis
                for analysis in trial["Analysis"]:
                    # Get arguments
                    args = []
                    for arg in analysis["analysis arguments"]:
                        try:
                            arg = locals()[arg]
                        except KeyError:
                            arg = arg
                        args.append(arg)
                    args.append(f"Scaffold Point: {configuration}")

                    print(f"Running analysis {analysis['analysis id']} for model: {model_name}")
                    chosen_module = __import__(analysis["analysis script"], fromlist=[''])
                    chosen_function = getattr(chosen_module, analysis["analysis function"])

                    chosen_function(*args)

                if "Delete Data" in trial:
                    if trial["Delete Data"]:
                        for file in new_data_files:
                            print(f"Deleting files {file}")
                            os.remove(file)
                        new_data_files = []

                complete = False

    def run_split_assay_mode(self, index, trial, running_jobs, memory_fraction, to_delete):
        # Run all assays, up to the point of the split.
        complete = False
        epsilon, total_steps, episode_number, configuration = self.get_saved_parameters(trial)

        new_job = self.get_new_job(trial, total_steps, episode_number, memory_fraction, epsilon, configuration)
        if new_job is not None:
            running_jobs[str(index)] = new_job
            running_jobs[str(index)].start()
            print(f"Starting {trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']}")
        else:
            print("New job failed")

        while len(running_jobs.keys()) > self.parallel_jobs - 1 and to_delete is None:
            for process in running_jobs.keys():
                if running_jobs[process].is_alive():
                    pass
                else:
                    to_delete = process
                    running_jobs[str(index)].join()
                    print(f"{trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']} Pre-Split Complete")
                    complete = True

        # Run again, twice from split point, with random seed set
        if complete:
            if to_delete is not None:
                del running_jobs[to_delete]
                to_delete = None

            complete = False
            # Second (modified) trial completion
            trial["Run Index"] = "Modified-Completion"
            new_job = self.get_new_job(trial, total_steps, episode_number, memory_fraction, epsilon,
                                       configuration)
            if new_job is not None:
                running_jobs[str(index)] = new_job
                running_jobs[str(index)].start()
                print(f"Starting {trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']}")
            else:
                print("New job failed")

            while len(running_jobs.keys()) > self.parallel_jobs - 1 and to_delete is None:
                for process in running_jobs.keys():
                    if running_jobs[process].is_alive():
                        pass
                    else:
                        to_delete = process
                        running_jobs[str(index)].join()
                        print(
                            f"{trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']} Post-Split Modified Complete")
                        complete = True
            to_delete = None

        if complete:
            if to_delete is not None:
                del running_jobs[to_delete]
                to_delete = None

            # First (normal) trial completion
            trial["Run Index"] = "Original-Completion"
            new_job = self.get_new_job(trial, total_steps, episode_number, memory_fraction, epsilon,
                                       configuration)
            if new_job is not None:
                running_jobs[str(index)] = new_job
                running_jobs[str(index)].start()
                print(f"Starting {trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']}")
            else:
                print("New job failed")

            while len(running_jobs.keys()) > self.parallel_jobs - 1 and to_delete is None:
                for process in running_jobs.keys():
                    if running_jobs[process].is_alive():
                        pass
                    else:
                        to_delete = process
                        running_jobs[str(index)].join()
                        print(
                            f"{trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']} Post-Split Original Complete")
                        complete = True
            to_delete = None

    def run_training(self, index, trial, running_jobs, memory_fraction, to_delete):
        epsilon, total_steps, episode_number, configuration = self.get_saved_parameters(trial)
        new_job = self.get_new_job(trial, total_steps, episode_number, memory_fraction, epsilon, configuration)
        if new_job is not None:
            running_jobs[str(index)] = new_job
            running_jobs[str(index)].start()
            print(f"Starting {trial['Model Name']} {trial['Trial Number']}, {trial['Run Mode']}")
        else:
            print("New job failed")

        while len(running_jobs.keys()) > self.parallel_jobs - 1 and to_delete is None:
            for process in running_jobs.keys():
                if running_jobs[process].is_alive():
                    pass
                else:
                    to_delete = process
                    running_jobs[str(index)].join()

    def run_priority_loop(self):
        """
        Executes the trials in the required order.
        :return:
        """
        memory_fraction = 0.99 / self.parallel_jobs
        running_jobs = {}
        to_delete = None

        for index, trial in enumerate(self.priority_ordered_trials):
            if to_delete is not None:
                del running_jobs[to_delete]
                to_delete = None
            elif trial["Run Mode"] == "Assay-Analysis-Across-Scaffold":
                self.run_analysis_across_scaffold(index, trial, running_jobs, memory_fraction, to_delete)
            elif trial["Run Mode"] == "Split-Assay":
                self.run_split_assay_mode(index, trial, running_jobs, memory_fraction, to_delete)
            else:
                self.run_training(index, trial, running_jobs, memory_fraction, to_delete)
