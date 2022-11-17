import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def get_fish_prey_vectors_proximity(data, proximity=15):
    fish_pos = data["fish_position"]
    prey_positions = data["prey_positions"]

    fish_pos = np.expand_dims(fish_pos, 1)
    vectors = fish_pos - prey_positions
    distances = (vectors[:, :, 0] ** 2 + vectors[:, :, 1] ** 2) ** 0.5
    within_range = distances < proximity
    return vectors[within_range]


def show_fish_prey_location_relative(model_name, assay_config, assay_id, n):
    """Displays cloud of fish-prey locations - with the aim of seeing whether or not the fluid displacement in model
    prevents fish correctly positioning prey."""
    f_p_vectors_compiled = []
    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        f_p_vectors = get_fish_prey_vectors_proximity(d)
        if len(f_p_vectors) > 0:
            f_p_vectors_compiled.append(f_p_vectors)
    f_p_vectors_compiled = np.concatenate(f_p_vectors_compiled)
    f_p_vectors_compiled = np.absolute(f_p_vectors_compiled)

    fig, ax = plt.subplots(figsize=(10, 10))

    plt.scatter(f_p_vectors_compiled[:, 0], f_p_vectors_compiled[:, 1])

    resolution = 10
    scale_bar = AnchoredSizeBar(ax.transData,
                                resolution, '1mm', 'lower left',
                                pad=1,
                                color='red',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16}
                                )
    ax.add_artist(scale_bar)

    plt.show()


if __name__ == "__main__":
    show_fish_prey_location_relative("dqn_new-2", "Behavioural-Data-Free", "Naturalistic", 20)


