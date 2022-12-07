import os

import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v1 as tf
import tensorflow
tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)

"""
Organisation of events:
   - Each file is made up of x of Event objects. 
   - I think each of these is an episode.

Event objects have important attributes:
   - step - int
   - wall_time - float
   - summary

summary has the following structure:
 value { 
    tag: str
    simple_value: whatever
}

"""


def load_all_log_data(model_name):
    available_tags = {}
    log_path = f"../../Training-Output/{model_name}/logs/"
    file_names = [f for f in os.listdir(log_path) if os.path.isfile(os.path.join(log_path, f))]

    for file in file_names:
        try:
            for i, summary in enumerate(tf.train.summary_iterator(f"{log_path}/{file}")):
                step = summary.step
                for v in summary.summary.value:
                    tag = v.tag
                    simple_value = v.simple_value
                    if tag in available_tags.keys():
                        available_tags[tag].append([step, simple_value])
                    else:
                        available_tags[tag] = [[step, simple_value]]
        except tensorflow.python.framework.errors_impl.DataLossError:
            print(f"{model_name} - Data loss")
    return available_tags


def order_metric_data(data):
    """Data is in format (t, 2) where t is number of timepoints sampled. Each point is [step, value]."""
    data = np.array(data)
    data = data[data[:, 0].argsort()]
    return data


if __name__ == "__main__":
    log_data = load_all_log_data("dqn_scaffold_28-1")
    rewards = np.array(log_data["episode reward"])
    plt.scatter(rewards[:, 0], rewards[:, 1])
    plt.show()

