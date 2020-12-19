import pymunk
import pymunk.pygame_util
import pygame

import numpy as np

space = pymunk.Space()


# space.gravity = 0, -900

def info(body):
    print(f'm={body.mass:.0f} moment={body.moment:.0f}')
    cg = body.center_of_gravity
    print(cg.x, cg.y)


class Box:
    def __init__(self, p0=(10, 10), p1=(1000, 700), d=2):
        x0, y0 = p0
        x1, y1 = p1
        pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            segment = pymunk.Segment(space.static_body, pts[i], pts[(i + 1) % 4], d)
            segment.elasticity = 1
            segment.friction = 1
            space.add(segment)


class Polygon:
    def __init__(self, pos, vertices, density=0.1):
        self.body = pymunk.Body(1, 100)
        self.body.position = pos

        shape = pymunk.Poly(self.body, vertices)
        shape.density = 0.1
        shape.elasticity = 1
        space.add(self.body, shape)


class Rectangle:
    def __init__(self, pos, size=(80, 50)):
        self.body = pymunk.Body()
        self.body.position = pos

        shape = pymunk.Poly.create_box(self.body, size)
        shape.density = 0.1
        shape.elasticity = 1
        shape.friction = 1
        space.add(self.body, shape)


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1000, 700))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    pygame.image.save(self.screen, 'shape.png')

            self.screen.fill((220, 220, 220))
            space.debug_draw(self.draw_options)
            pygame.display.update()
            space.step(0.01)

        pygame.quit()


if __name__ == '__main__':
    Box()
    body = pymunk.Body(mass=1, moment=1000)
    body.position = (900, 100)
    body.angle = np.random.random() * 2 * np.pi
    # body.apply_impulse_at_local_point((100, 0))

    # Mouth
    mouth = pymunk.Circle(body, 15, offset=(0, 0))
    mouth.color = (1, 0, 1)
    mouth.elasticity = 1.0
    mouth.collision_type = 3

    # Head
    head = pymunk.Circle(body, 30, offset=(30, 0))
    head.color = (0, 1, 0)
    head.elasticity = 1.0
    head.collision_type = 3

    tail_coordinates = ((30, 0), (30, 30), (120, 0),
                        (30, -30))
    tail = pymunk.Poly(body, tail_coordinates)
    tail.color = (0, 1, 0)
    tail.elasticity = 1.0
    tail.collision_type = 3

    space.add(body, mouth, head, tail)
    App().run()