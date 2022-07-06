import os
import numpy as np

from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files


class DataIndexServiceDiscrete:
    """Idea is that this provides an easily interpretable wrapper for applying all the standard scripts (already
    existing in other Analysis directories), in a more standardised, interpretable way."""

    def __init__(self, model_name):
        naturalistic_preset = "Behavioural-Data-Free"
        naturalistic_suffix = "Naturalistic-"  # Try loading for all within large range.

        self.model_name = model_name
        self.environmental_config, self.learning_config, _, _, _ = load_configuration_files(self.model_name)

        # Auto-load from all file presets.
        self.naturalistic_trial_data = self.load_all_data(naturalistic_preset, naturalistic_suffix)
        self.flattened_naturalistic_trial_data, self.naturalistic_trial_lengths = self._flatten_data_list(self.naturalistic_trial_data)

        # Create output data location
        if __name__ == "__main__":
            if not os.path.exists(f"../../Analysis/Data/Figures/{model_name}/"):
                os.makedirs(f"../../Analysis/Data/Figures/{model_name}/")
            self.figure_save_location = f"../../Analysis/Data/Figures/{model_name}/"
        else:
            if not os.path.exists(f"./Analysis/Data/Figures/{model_name}/"):
                os.makedirs(f"./Analysis/Data/Figures/{model_name}/")
            self.figure_save_location = f"./Analysis/Data/Figures/{model_name}/"

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
        ...

    def produce_neural_summary_display(self):
        ...


class DataIndexServiceContinuous(DataIndexServiceDiscrete):

    def __init__(self, model_name):
        super().__init__(model_name)


if __name__ == "__main__":
    DataIndexServiceDiscrete("dqn_scaffold_23-1")
