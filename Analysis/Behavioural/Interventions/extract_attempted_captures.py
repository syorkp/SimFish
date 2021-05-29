from Analysis.load_data import load_data
from Analysis.Behavioural.New.display_action_sequences import display_all_sequences_capture, get_capture_sequences


def extract_attempted_captures(data):
    prey_timestamps = []
    sensing_distance = 200
    for i, p in enumerate(data["position"]):
        for prey in data["prey_positions"][i]:
            sensing_area = [[p[0] - sensing_distance,
                             p[0] + sensing_distance],
                            [p[1] - sensing_distance,
                             p[1] + sensing_distance]]
            near_prey = sensing_area[0][0] <= prey[0] <= sensing_area[0][1] and \
                        sensing_area[1][0] <= prey[1] <= sensing_area[1][1]
            if near_prey:
                prey_timestamps.append(i)
                break
    successful_captures = sum(data["consumed"])
    x = True


extract_attempted_captures(load_data("new_even_prey_ref-4", "Ablation-Test-15-Centre-even_prey_only", "Ablated-0-1"))

# cs = get_capture_sequences("new_even_prey_ref-4", "Ablation-Test-Predator_Only-behavioural_data", "Random-Control", 12)
# display_all_sequences_capture(cs[:26])
x = True
