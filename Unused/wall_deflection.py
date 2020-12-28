def calculate_angle_of_adjustment(self, angle_of_incidence):
    if angle_of_incidence % np.pi < np.pi / 2:
        return np.pi - (2 * angle_of_incidence)
    else:
        return -(2 * angle_of_incidence) - np.pi / 2


def touch_edge(self, arbiter, space, data):
    # TODO: Fix, currently not producing desired angles, or deflection.
    self.fish.body.velocity = (0, 0)
    current_position = self.fish.body.position
    print("Collided!")
    inc = None
    # Decide which wall and calculate angle of incidence, update position accordingly.
    if current_position[0] < 10:  # Wall d
        print("Collided with wall D")
        inc = np.pi - self.fish.body.angle
        current_position[0] = 80  # TODO: Make these adjustments with respect to fish size.
    elif current_position[0] > self.env_variables['width'] - 10:  # wall b
        print("Collided with wall B")
        inc = np.pi / 2 - self.fish.body.angle
        current_position[0] = self.env_variables['width'] - 80
    if current_position[1] < 10:  # wall a
        print("Collided with wall A")
        inc = (2 * np.pi) - self.fish.body.angle
        current_position[1] = 80
    elif current_position[1] > self.env_variables['height'] - 10:  # wall c
        print("Collided with wall C")
        inc = self.fish.body.angle
        current_position[1] = self.env_variables['height'] - 80

    self.fish.body.position = current_position
    if inc is not None:
        print(inc)
        adj = self.calculate_angle_of_adjustment(inc)
    else:
        return True

    self.fish.body.angle += adj

    self.fish.body.velocity = (0, 0)

    if self.fish.body.angle < np.pi:
        self.fish.body.angle += np.pi
    else:
        self.fish.body.angle -= np.pi
    self.fish.touched_edge = True
    return True