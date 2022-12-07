from Analysis.load_data import load_data




def build_positional_model(model_name, assay_config, assay_id, n):
    compiled_prey_positions = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        compiled_prey_positions.append(data["prey_positions"])
    # TODO: Reformat data into 10-1 X y trials.
    # TODO: Build model based on data.


if __name__ == "__main__":
    build_positional_model("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20)
    # TODO: Improve by taking into account the position of the fish during training
