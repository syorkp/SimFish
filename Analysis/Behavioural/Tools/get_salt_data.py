from Analysis.load_data import load_data


def get_salt_data(model_name, assay_config, assay_id, n):
    fish_positions = []
    fish_orientations = []
    salt_source_locations = []
    salt_concentrations = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        fish_positions.append(data["fish_position"])
        fish_orientations.append(data["fish_angle"])
        salt_source_locations.append(data["salt_location"] for i in range(len(data["fish_angle"])))
        salt_concentrations.append(data["salt"])

    return fish_positions, fish_orientations, salt_source_locations, salt_concentrations
