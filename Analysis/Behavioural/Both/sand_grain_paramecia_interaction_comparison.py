from Analysis.load_data import load_data


def get_num_engagement_sg_p(data, range_for_engagement_poss, range_for_engagement_def):
    """Returns the numbers of possible and actual engagements with each feature"""
    num_sg_interactions = 0
    num_sg_poss_interactions = 0

    num_prey_interactions = 0
    num_prey_poss_interactions = 0

    for step in range(data["action"].shape[0]):
        fish_position = data["fish_position"][step]
        prey_positions = data["prey_positions"][step]
        sand_grain_positions = data["sand_grain_positions"][step]

        fp_vectors = prey_positions - fish_position
        fsg_vectors = sand_grain_positions - fish_position

        fp_distances = (fp_vectors[:, 0] ** 2 + fp_vectors[:, 1] ** 2) ** 0.5
        fsg_distances = (fsg_vectors[:, 0] ** 2 + fsg_vectors[:, 1] ** 2) ** 0.5
        x = True


def compute_approach_probability_sg_p(model_name, assay_config, assay_id, n):
    for i in range(1, n+1):
        d = load_data(model_name, assay_config, f"{assay_id}-{i}")
        get_num_engagement_sg_p(d, range_for_engagement_poss=100, range_for_engagement_def=5)
        x = True




if __name__ == "__main__":
    compute_approach_probability_sg_p("dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic", 10)

