import numpy as np
import matplotlib.pyplot as plt
import copy
from skimage.transform import resize, rescale

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Services.base_service import BaseService
from Services.Testing.visualisation_board import VisualisationBoard


class EnvTestingService(BaseService):

    def __init__(self, config_name, continuous_actions):
        super().__init__(model_name="Test",
                         trial_number=1,
                         total_steps=0,
                         episode_number=0,
                         monitor_gpu=False,
                         using_gpu=False,
                         memory_fraction=1,
                         config_name=config_name,
                         continuous_actions=continuous_actions,
                         monitor_performance=False
                         )

        self.current_configuration_location = f"./Configurations/Assay-Configs/{self.config_name}"
        _, self.environment_params = self.load_configuration_files()

        # Create environment
        if self.continuous_actions:
            self.simulation = ContinuousNaturalisticEnvironment(self.environment_params,
                                                                False,
                                                                num_actions=12,
                                                                )
        else:
            self.simulation = DiscreteNaturalisticEnvironment(self.environment_params,
                                                              False,
                                                              num_actions=12,
                                                              )

        # Create visualisation board
        self.visualisation_board = VisualisationBoard(self.environment_params["arena_width"],
                                                      self.environment_params["arena_height"],
                                                      light_gradient=self.environment_params["light_gradient"])
        self.board_fig, self.ax_board = plt.subplots()
        self.board_image = plt.imshow(np.zeros((self.environment_params['arena_height']*4, self.environment_params['arena_width']*4, 3)))
        plt.ion()
        plt.show()

    def episode_loop(self):
        ended = False
        while not ended:
            print("Enter action: ")
            if self.continuous_actions:
                impulse = input()
                angle = input()

                impulse = float(impulse)
                angle = float(angle)

                action = [impulse, angle]
            else:
                action = input()
                action = int(action)

            if action == 99:
                plt.rcParams['figure.figsize'] = [10, 10]
                plt.savefig("Fig")
            else:
                self.simulation.simulation_step(action)
                self.update_drawing()

    def update_drawing(self):
        self.visualisation_board.erase_visualisation()
        self.draw_shapes()
        self.visualisation_board.apply_light(dark_col=int(self.environment_params["dark_light_ratio"] * self.environment_params["arena_width"]),
                                             dark_gain=self.environment_params["dark_gain"],
                                             light_gain=self.environment_params["light_gain"])
        frame = self.output_frame(scale=4) / 255.
        self.board_image.set_data(frame / np.max(frame))
        plt.pause(0.000001)

    def draw_shapes(self):
        scaling_factor = 1500 / self.environment_params["arena_width"]
        prey_size = self.environment_params['prey_radius_visualisation']/scaling_factor

        fish_body_colour = (1 - self.simulation.fish.energy_level, self.simulation.fish.energy_level, 0)

        self.visualisation_board.fish_shape(self.simulation.fish.body.position,
                                            self.environment_params['fish_mouth_radius'],
                                            self.environment_params['fish_head_radius'],
                                            self.environment_params['fish_tail_length'],
                                            self.simulation.fish.mouth.color,
                                            fish_body_colour,
                                            self.simulation.fish.body.angle)

        if len(self.simulation.prey_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.simulation.prey_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.simulation.prey_bodies])).astype(int)
            rrs, ccs = self.visualisation_board.multi_circles(px, py, prey_size)
            rrs = np.clip(rrs, 0, self.environment_params["arena_width"]-1)
            ccs = np.clip(ccs, 0, self.environment_params["arena_height"]-1)

            try:
                self.visualisation_board.db_visualisation[rrs, ccs] = self.simulation.prey_shapes[0].color
            except IndexError:
                print(f"Index Error for: PX: {max(rrs.flatten())}, PY: {max(ccs.flatten())}")
                if max(rrs.flatten()) > self.environment_params['arena_height']:
                    lost_index = np.argmax(py)
                elif max(ccs.flatten()) > self.environment_params['arena_width']:
                    lost_index = np.argmax(px)
                else:
                    lost_index = 0
                    print(f"Fix needs to be tuned: PX: {max(px)}, PY: {max(py)}")
                self.simulation.prey_bodies.pop(lost_index)
                self.simulation.prey_shapes.pop(lost_index)
                self.draw_shapes()

        if len(self.simulation.sand_grain_bodies) > 0:
            px = np.round(np.array([pr.position[0] for pr in self.simulation.sand_grain_bodies])).astype(int)
            py = np.round(np.array([pr.position[1] for pr in self.simulation.sand_grain_bodies])).astype(int)
            rrs, ccs = self.visualisation_board.multi_circles(px, py, prey_size)

            try:
                if visualisation:
                    self.visualisation_board.db_visualisation[rrs, ccs] = self.simulation.sand_grain_shapes[0].color
                else:
                    self.visualisation_board.db[rrs, ccs] = self.simulation.sand_grain_shapes[0].color
            except IndexError:
                print(f"Index Error for: RRS: {max(rrs.flatten())}, CCS: {max(ccs.flatten())}")
                if max(rrs.flatten()) > self.environment_params['arena_width']:
                    lost_index = np.argmax(px)
                elif max(ccs.flatten()) > self.environment_params['arena_height']:
                    lost_index = np.argmax(py)
                else:
                    lost_index = 0
                    print(f"Fix needs to be tuned: PX: {max(px)}, PY: {max(py)}")
                self.simulation.sand_grain_bodies.pop(lost_index)
                self.simulation.sand_grain_shapes.pop(lost_index)
                self.draw_shapes()

        for i, pr in enumerate(self.simulation.predator_bodies):
            self.visualisation_board.circle(pr.position, self.environment_params['predator_radius'],
                                            self.simulation.predator_shapes[i].color, True)

        if self.simulation.predator_body is not None:
            self.visualisation_board.circle(self.simulation.predator_body.position, self.environment_params['predator_radius'],
                              self.simulation.predator_shape.color, True)

        # if self.environment_params["salt"] and self.environment_params["max_salt_damage"] > 0:
        #     self.visualisation_board.show_salt_location(self.simulation.salt_location)

    def output_frame(self, internal_state=np.array([[0, 0, 0]]), scale=0.25):
        # Adjust scale for larger environments

        # Saving mask frames (for debugging)

        arena = copy.copy(self.visualisation_board.db_visualisation * 255.0)

        arena[0, :, 0] = np.ones(self.environment_params['arena_width']) * 255
        arena[self.environment_params['arena_height'] - 1, :, 0] = np.ones(self.environment_params['arena_width']) * 255
        arena[:, 0, 0] = np.ones(self.environment_params['arena_height']) * 255
        arena[:, self.environment_params['arena_width'] - 1, 0] = np.ones(self.environment_params['arena_height']) * 255

        empty_green_eyes = np.zeros((20, self.environment_params["arena_width"], 1))
        left_photons = self.simulation.fish.readings_to_photons(self.simulation.fish.left_eye.readings)
        right_photons = self.simulation.fish.readings_to_photons(self.simulation.fish.right_eye.readings)

        left_eye = resize(np.reshape(left_photons, (1, self.simulation.fish.left_eye.observation_size, 3)) * (
                255 / 100), (20, self.environment_params['arena_width'] / 2 - 50))
        right_eye = resize(np.reshape(right_photons, (1, self.simulation.fish.right_eye.observation_size, 3)) * (
                255 / 100), (20, self.environment_params['arena_width'] / 2 - 50))
        eyes = np.hstack((left_eye, np.zeros((20, 100, 3)), right_eye))
        eyes = np.concatenate((eyes[:, :, :1], empty_green_eyes, eyes[:, :, 1:2]),
                                  axis=2)  # Note removes second red channel.

        frame = np.vstack((arena, np.zeros((50, self.environment_params['arena_width'], 3)), eyes))

        this_ac = np.zeros((20, self.environment_params['arena_width'], 3))
        this_ac[:, :, 0] = resize(internal_state, (20, self.environment_params['arena_width']), anti_aliasing=False,
                                  order=0) * 255
        this_ac[:, :, 1] = resize(internal_state, (20, self.environment_params['arena_width']), anti_aliasing=False,
                                  order=0) * 255
        this_ac[:, :, 2] = resize(internal_state, (20, self.environment_params['arena_width']), anti_aliasing=False,
                                  order=0) * 255

        frame = np.vstack((frame, np.zeros((20, self.environment_params['arena_width'], 3)), this_ac))

        frame = rescale(frame, scale, multichannel=True, anti_aliasing=True)

        return frame




