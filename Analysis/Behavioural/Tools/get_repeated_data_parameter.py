from Analysis.load_data import load_data


def get_parameter_across_trials(model_name, assay_config, assay_id, n, desired_parameter="fish_position"):
    all_data = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_data.append(data[desired_parameter])
    return all_data
