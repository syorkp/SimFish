
def get_salt_data(model_name, assay_config, assay_id, n):
    fish_positions = []
    fish_orientations = []
    salt_source_locations = []
    salt_concentrations = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        all_capture_sequences = all_capture_sequences + extract_consumption_action_sequences(data)[0]
    return all_capture_sequences
