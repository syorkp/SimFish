
class Environment:

    def __init__(self):
        self.space = None
        self.complex_predator_size = None
        self.complex_predator_body = None
        self.complex_predator_shape = None

    def remove_complex_predator(self):
        if self.complex_predator_body is not None:
            self.space.remove(self.complex_predator_shape, self.complex_predator_shape.body)
            self.complex_predator_shape = None
            self.complex_predator_body = None
            self.complex_predator_size = None
        else:
            pass

    def create_complex_predator(self, position=None, size=None):
        self.remove_complex_predator()

        self.complex_predator_body = pymunk.Body(self.env_variables['predator_mass'], self.env_variables['predator_inertia'])

        if size is None:
            self.complex_predator_shape = pymunk.Circle(self.complex_predator_body, self.env_variables['predator_size'])
            self.complex_predator_size = self.env_variables['predator_size']
        else:
            self.complex_predator_shape = pymunk.Circle(self.complex_predator_body, size)
            self.complex_predator_size = size

        if position is None:
            distance_from_fish = 30  # TODO: make part of configuration parameters
            fish_position = self.fish.body.position
            angle_from_fish = random.randint(0, 360)
            angle_from_fish = math.radians(angle_from_fish / math.pi)
            dy = distance_from_fish * math.cos(angle_from_fish)
            dx = distance_from_fish * math.sin(angle_from_fish)
            self.complex_predator_body.position = (fish_position[0] + dx, fish_position[1] + dy)
        else:
            self.complex_predator_body.position = position

        self.complex_predator_shape.color = (0, 0, 1)
        self.complex_predator_shape.collision_type = 5

        self.space.add(self.complex_predator_body, self.complex_predator_shape)

    def grow_complex_predator(self):
        size_limit = 100
        increment = 0.03  # TODO: Add this and the above to configurations
        if self.complex_predator_size < size_limit:
            self.create_complex_predator(self.complex_predator_body.position, self.complex_predator_size + increment)
        else:
            self.remove_complex_predator()

