
from Analysis.load_model_config import load_configuration_files
from Environment.Fish.fish import Fish
from Tools.drawing_board_new import NewDrawingBoard


learning_params, env_variables, n, b, c = load_configuration_files("dqn_scaffold_14-1")
db = NewDrawingBoard(env_variables["width"],
                     env_variables["height"],
                     env_variables["decay_rate"],
                     env_variables["uv_photoreceptor_rf_size"],
                     False,
                     False)
fish = Fish(db, env_variables, 0.0, True, True, False)