from Analysis.Behavioural.Discrete.SpatialDensity.show_spatial_density_discrete import get_all_density_plots_all_subsets


if __name__ == "__main__":
    # Getting positions for several steps prior.
    # get_all_density_plots_all_subsets(f"dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20, return_objects=False, steps_prior=9)

    # Getting positions for "predicted" prey locations
    get_all_density_plots_all_subsets(f"dqn_beta-1", "Behavioural-Data-Free", "Naturalistic", 20,
                                      return_objects=False, steps_prior=0, position_predictive=True)



