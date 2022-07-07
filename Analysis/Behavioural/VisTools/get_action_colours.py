"""To be for discrete and continuous."""


def create_color_map(actions):
    colors = []
    for action in actions:
        if action == "RT Right" or action == "RT Left":
            colors.append("green")
        elif action == "sCS":
            colors.append("red")
        elif action == "J-turn Left" or action == "J-turn Right":
            colors.append("yellow")
        else:
            colors.append("purple")

    return colors
