import numpy as np

from Analysis.load_data import load_data


def get_num_engagement_sg_p(data, range_for_engagement_poss, range_for_engagement_def, sparsen_interactions):
    """Returns the numbers of possible and actual engagements with each feature"""
    num_sg_interactions = 0
    num_sg_poss_interactions = 0

    num_prey_interactions = 0
    num_prey_poss_interactions = 0

    fish_position = np.expand_dims(data["fish_position"], 1)
    prey_positions = data["prey_positions"]
    sand_grain_positions = data["sand_grain_positions"]

    fp_vectors = prey_positions - fish_position
    fsg_vectors = sand_grain_positions - fish_position

    fp_distances = (fp_vectors[:, :, 0] ** 2 + fp_vectors[:, :, 1] ** 2) ** 0.5
    fsg_distances = (fsg_vectors[:, :, 0] ** 2 + fsg_vectors[:, :, 1] ** 2) ** 0.5

    fp_within_poss_range = (fp_distances < range_for_engagement_poss) * 1
    fp_within_int_range = (fp_distances < range_for_engagement_def) * 1

    fsg_within_poss_range = (fsg_distances < range_for_engagement_poss) * 1
    fsg_within_int_range = (fsg_distances < range_for_engagement_def) * 1

    # Make the interaction matrices sparse
    if sparsen_interactions:
        for step in reversed(range(fp_within_poss_range.shape[0])):
            if step == fp_within_poss_range.shape[0]:
                pass
            else:
                fp_within_poss_range[step] *= (fp_within_poss_range[step-1] == 0) * 1
                fp_within_int_range[step] *= (fp_within_int_range[step-1] == 0) * 1
                fsg_within_poss_range[step] *= (fsg_within_poss_range[step-1] == 0) * 1
                fsg_within_int_range[step] *= (fsg_within_int_range[step-1] == 0) * 1
    # fp_within_poss_range_time_shifted = np.concatenate((fp_within_poss_range[1:], np.ones((1, fp_within_poss_range.shape[1])).astype(int)), axis=0)
    # fp_within_poss_range_sparse = fp_within_poss_range * fp_within_poss_range_time_shifted

    num_prey_interactions += np.sum(fp_within_int_range)
    num_prey_poss_interactions += np.sum(fp_within_poss_range)
    num_sg_interactions += np.sum(fsg_within_int_range)
    num_sg_poss_interactions += np.sum(fsg_within_poss_range)

    return num_prey_interactions, num_prey_poss_interactions, num_sg_interactions, num_sg_poss_interactions


def compute_approach_probability_sg_p(model_name, assay_config, assay_id, n):
    """Computes the probability of approach for SGs and P - NOTE! This metric will be biased when fish rest near a
    feature - will show up as high probability."""

    num_prey_interactions_compiled = 0
    num_prey_poss_interactions_compiled = 0
    num_sg_interactions_compiled = 0
    num_sg_poss_interactions_compiled = 0
    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        num_prey_interactions, num_prey_poss_interactions, num_sg_interactions, num_sg_poss_interactions = \
            get_num_engagement_sg_p(d, range_for_engagement_poss=100, range_for_engagement_def=30, sparsen_interactions=True)
        num_prey_interactions_compiled += num_prey_interactions
        num_prey_poss_interactions_compiled += num_prey_poss_interactions
        num_sg_interactions_compiled += num_sg_interactions
        num_sg_poss_interactions_compiled += num_sg_poss_interactions

    if num_prey_poss_interactions_compiled == 0:
        prey_approach_probability = 0
    else:
        prey_approach_probability = num_prey_interactions_compiled / num_prey_poss_interactions_compiled

    if num_sg_poss_interactions_compiled == 0:
        sand_grain_approach_probability = 0
    else:
        sand_grain_approach_probability = num_sg_interactions_compiled / num_sg_poss_interactions_compiled

    return prey_approach_probability, sand_grain_approach_probability


if __name__ == "__main__":
    prey_approach_probability, sand_grain_approach_probability = \
        compute_approach_probability_sg_p("dqn_scaffold_33-2", "Behavioural-Data-Free", "Naturalistic", 20)
