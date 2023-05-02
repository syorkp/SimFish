

def get_action_name(action_num):
    if action_num == 0:
        action_name = "Slow2"
    elif action_num == 2:
        action_name = "RT Right"
    elif action_num == 1:
        action_name = "RT Left"
    elif action_num == 3:
        action_name = "sCS"
    elif action_num == 5:
        action_name = "J-turn Right"
    elif action_num == 4:
        action_name = "J-turn Left"
    elif action_num == 6:
        action_name = "Rest"
    elif action_num == 8:
        action_name = "SLC Right"
    elif action_num == 7:
        action_name = "SLC Left"
    elif action_num == 9:
        action_name = "AS"
    elif action_num == 11:
        action_name = "J-turn 2 Right"
    elif action_num == 10:
        action_name = "J-turn 2 Left"
    else:
        action_name = "None"
    return action_name


def get_action_name_unlateralised(action_num):
    if action_num == 0:
        action_name = "Slow2"
    elif action_num == 1:
        action_name = "RT"
    elif action_num == 2:
        action_name = "RT"
    elif action_num == 3:
        action_name = "sCS"
    elif action_num == 4:
        action_name = "J-turn"
    elif action_num == 5:
        action_name = "J-turn"
    elif action_num == 6:
        action_name = "Rest"
    elif action_num == 7:
        action_name = "SLC"
    elif action_num == 8:
        action_name = "SLC"
    elif action_num == 9:
        action_name = "AS"
    else:
        action_name = "None"
    return action_name