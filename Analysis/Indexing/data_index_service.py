import os
import numpy as np

from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files

from Analysis.Behavioural.Tools.BehavLabels.extract_capture_sequences import extract_consumption_action_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import extract_no_prey_stimuli_sequences, extract_exploration_action_sequences_with_positions
from Analysis.Behavioural.Tools.BehavLabels.extract_escape_sequences import extract_escape_action_sequences_with_positions

from Analysis.Behavioural.VisTools.show_action_sequence_block import display_all_sequences
from Analysis.Behavioural.Discrete.SpatialDensity.show_spatial_density_discrete import get_all_density_plots
from Analysis.Behavioural.TurnChains.turning_analysis_discrete import get_cumulative_switching_probability_plot
from Analysis.Behavioural.Both.phototaxis import plot_light_dark_occupancy_kdf, plot_luminance_driven_choice


class DataIndexServiceDiscrete:
    """Idea is that this provides an easily interpretable wrapper for applying all the standard scripts (already
    existing in other Analysis directories), in a more standardised, interpretable way."""

    def __init__(self, model_name):
        naturalistic_preset = "Behavioural-Data-Free"
        naturalistic_suffix = "Naturalistic-"  # Try loading for all within large range.

        self.model_name = model_name
        self.learning_config, self.environmental_config, _, _, _ = load_configuration_files(self.model_name)

        # Auto-load from all file presets.
        self.naturalistic_trial_data = self.load_all_data(naturalistic_preset, naturalistic_suffix)
        self.flattened_naturalistic_trial_data, self.naturalistic_trial_lengths = self._flatten_data_list(self.naturalistic_trial_data)

        # Create output data location
        if __name__ == "__main__":
            self.figure_save_location = f"../../Assay-Output/{model_name}/Figures"
        else:
            self.figure_save_location = f"./Assay-Output/{model_name}/Figures"

        if not os.path.exists(self.figure_save_location):
            os.makedirs(f"{self.figure_save_location}")
            os.makedirs(f"{self.figure_save_location}/Spatial-Density-Plots")
            os.makedirs(f"{self.figure_save_location}/Behavioural-Metrics")
            os.makedirs(f"{self.figure_save_location}/Prey-Capture")
            os.makedirs(f"{self.figure_save_location}/Exploration")
            os.makedirs(f"{self.figure_save_location}/Predator-Avoidance")
            os.makedirs(f"{self.figure_save_location}/Phototaxis")
            os.makedirs(f"{self.figure_save_location}/Salt")

        # Extract main types of sequences
        self.capture_sequences, self.exploration_sequences_1, self.exploration_sequences_2, self.escape_sequences = [], [], [], []
        self.extract_sequences_groups()

    @staticmethod
    def _flatten_data_list(data_list):
         # TODO: Change to ensure 1D lists e.g. salt position are not being concatenated completely.
        flattened_data_dictionary = {}
        trial_lengths = []
        for i, key in enumerate(data_list[0].keys()):
            for j in range(len(data_list)):
                if i == 0:
                    trial_lengths.append(len(data_list[j][key]))
                if key in flattened_data_dictionary:
                    flattened_data_dictionary[key] = np.concatenate((flattened_data_dictionary[key], data_list[j][key]))
                else:
                    flattened_data_dictionary[key] = data_list[j][key]
        return flattened_data_dictionary, trial_lengths

    def load_all_data(self, assay_group, assay_id):
        current_index = 1
        data_list = []
        while True:
            try:
                data = load_data(self.model_name, assay_group, assay_id + str(current_index))
                data_list.append(data)
                current_index += 1
            except AttributeError:
                print(f"Nothing beyond {current_index}")
                break

        return data_list

    def extract_sequences_groups(self):
        for d in self.naturalistic_trial_data:
            self.capture_sequences += extract_consumption_action_sequences(d)[0]
            self.exploration_sequences_1 += extract_no_prey_stimuli_sequences(d)
            self.exploration_sequences_2 += extract_exploration_action_sequences_with_positions(d)[1]
            self.escape_sequences += extract_escape_action_sequences_with_positions(d)[1][0]

    def get_data_simple_condition(self, condition, null_alternatives=False, return_as_indexes=False, specific_key=False):
        """To get all data at timestamps where a condition is met - no checking over multiple timestamps.

        Uses the flattened data for this.

        Condition format - {"action": 1}/{consumed: True} TODO: Create method to turn non-readable conditions readable

        Conditions (and format):
        - action usage (specified as a range for continuous)
        - consumption
        - predator present
        - in light/dark
        - unsustainable salt
        - within a current
        - within x distance of a feature: wall, predator, prey.
        - more complex spatial relationships e.g. egocentric prey, at a given angle.

        null_alternatives - excludes data where other outstanding conditions are met e.g. if there are any other prey
        within understood visual range.
        return_as_indexes - returns not the final data, but the indexes within the flattened array that match condition,
        which is necessary for determining combinations of conditions.
        specific_key - option to return only a certain data kind e.g. action choice.
        """
        ...

    def get_data_combined_simple_conditions(self):
        ...

    def produce_behavioural_summary_display(self):
        """Initially, produce all the elements individually and save them as jpegs"""
        # Bin across energy states.
        # capture_sequences, energy_states_cs = get_capture_sequences_with_energy_state(model_name, assay_group,
        #                                                                               assay_name, n)
        # exploration_sequences, energy_states_ex = get_exploration_sequences_with_energy_state(model_name, assay_group, assay_name, n)
        # plot_energy_state_grouped_action_usage_from_data()

        # Display all spatial density plots
        get_all_density_plots(self.flattened_naturalistic_trial_data, self.figure_save_location + "/Spatial-Density-Plots")

        # Prey Capture
        display_all_sequences(self.capture_sequences[:100], save_figure=True,
                              figure_save_location=f"{self.figure_save_location}/Prey-Capture/Sequences.jpg")

        # Exploration
        display_all_sequences(self.exploration_sequences_1[:100], save_figure=True,
                              figure_save_location=f"{self.figure_save_location}/Exploration/Sequences1.jpg")
        display_all_sequences(self.exploration_sequences_2[:100], save_figure=True,
                              figure_save_location=f"{self.figure_save_location}/Exploration/Sequences2.jpg")
        get_cumulative_switching_probability_plot(self.exploration_sequences_1,
                                                  figure_save_location=f"{self.figure_save_location}/Exploration/Cumulative-Switching-Probabiolity-1.jpg")
        get_cumulative_switching_probability_plot(self.exploration_sequences_2,
                                                  figure_save_location=f"{self.figure_save_location}/Exploration/Cumulative-Switching-Probabiolity-2.jpg")

        # Predator Avoidance
        display_all_sequences(self.escape_sequences[:100], save_figure=True,
                              figure_save_location=f"{self.figure_save_location}/Predator-Avoidance/Sequences.jpg")

        # Phototaxis
        fish_position_data = self.flattened_naturalistic_trial_data["fish_position"]
        action_data = self.flattened_naturalistic_trial_data["action"]
        observation_data = self.flattened_naturalistic_trial_data["observation"]
        plot_light_dark_occupancy_kdf(fish_position_data, self.environmental_config, self.figure_save_location + "/Phototaxis/Light-Dark-Occupancy.jpg")
        plot_luminance_driven_choice(observation_data, action_data, fish_position_data, self.environmental_config,
                                     self.figure_save_location + "/Phototaxis/Luminance-Driven-Choice.jpg")

    def produce_neural_summary_display(self):
        ...


class DataIndexServiceContinuous(DataIndexServiceDiscrete):

    def __init__(self, model_name):
        super().__init__(model_name)


if __name__ == "__main__":
    d = DataIndexServiceDiscrete("dqn_scaffold_19-1")
    d.produce_behavioural_summary_display()
