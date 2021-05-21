import numpy as np
import matplotlib.pyplot as plt
import pymunk

from Environment.base_environment import BaseEnvironment
from Environment.Fish.fish import Fish
from Environment.Fish.tethered_fish import TetheredFish


class ControlledStimulusEnvironment(BaseEnvironment):
    """
    This version is made with only the fixed projection configuration in mind.
    As a result, doesnt have walls, and fish appears directly in the centfre of the environment.
    For this environment, the following stimuli are available: prey, predators.
    """

    def __init__(self, env_variables, stimuli, realistic_bouts, tethered=True, set_positions=False, moving=False,
                 random=False, reset_each_step=False, reset_interval=1, background=None, draw_screen=False):
        super().__init__(env_variables, draw_screen)

        if tethered:
            self.fish = TetheredFish(self.board, env_variables, self.dark_col, realistic_bouts)
        else:
            self.fish = Fish(self.board, env_variables, self.dark_col, realistic_bouts)
        self.space.add(self.fish.body, self.fish.mouth, self.fish.head, self.fish.tail)

        # TODO: Unify in future with other stimuli
        self.prey_positions = {}
        self.predator_positions = {}
        self.set_positions = set_positions
        self.random = random
        self.reset_at_interval = reset_each_step
        self.reset_interval = reset_interval

        # Whole environment measurements.
        board_height = env_variables["height"]
        board_width = env_variables["width"]

        # Wall coordinates
        self.wall_1_coordinates = [[0, 0], [0, board_height]]
        self.wall_2_coordinates = [[0, board_height], [board_width, board_height]]
        self.wall_3_coordinates = [[1, 1], [board_width,1]]
        self.wall_4_coordinates = [[board_width, 1], [board_width, board_height]]

        self.stimuli = stimuli

        self.stimuli_information = {stimulus: {} for stimulus in stimuli}

        self.create_walls()
        self.reset()

        if self.set_positions:
            self.create_positional_information(stimuli)
        else:
            if self.random:
                self.random_stimuli = stimuli
            else:
                self.unset_stimuli = stimuli
        self.moving_stimuli = moving

        self.edge_col = self.space.add_collision_handler(1, 3)
        self.edge_col.begin = self.touch_edge
        self.background = background
        self.pred_fish_col = self.space.add_collision_handler(3, 5)
        self.pred_fish_col.begin = self.no_collision
        self.prey_fish_col = self.space.add_collision_handler(3, 2)
        self.prey_fish_col.begin = self.no_collision

    def reset(self):
        super().reset()
        self.fish.body.position = (self.env_variables['width']/2, self.env_variables['height']/2)
        self.fish.body.angle = 0
        self.fish.body.velocity = (0, 0)
        self.create_stimuli(self.stimuli)

    def special_reset(self):
        self.fish.body.position = (self.env_variables['width'] / 2, self.env_variables['height'] / 2)
        self.fish.body.angle = 0
        self.fish.body.velocity = (0, 0)
        self.fish.hungry = 0

    def simulation_step(self, action, save_frames=False, frame_buffer=None, activations=None):
        if self.reset_at_interval and self.num_steps % self.reset_interval == 0:
            self.special_reset()
        if frame_buffer is None:
            frame_buffer = []
        self.fish.making_capture = False
        reward = self.fish.take_action(action)

        done = False

        self.fish.hungry += (1 - self.fish.hungry)*self.env_variables['hunger_inc_tau']
        self.fish.stress = self.fish.stress * self.env_variables['stress_compound']
        if self.predator_body is not None:
            self.fish.stress += 0.5

        # According to the projection general mode:
        if self.set_positions:
            self.update_stimuli()
        else:
            if self.random:
                self.update_random_stimuli()
            else:
                self.update_unset_stimuli()

        for micro_step in range(self.env_variables['phys_steps_per_sim_step']):
            self.space.step(self.env_variables['phys_dt'])
            if self.fish.touched_edge:
                self.fish.touched_edge = False
            if self.show_all:
                self.board.erase()
                self.draw_shapes()
                if self.draw_screen:
                    self.board_image.set_data(self.output_frame(activations, np.array([0, 0]), scale=0.5)/255.)
                    plt.pause(0.0001)

        self.fish.body.position = (self.env_variables['width'] / 2, self.env_variables['height'] / 2)
        self.fish.body.angle = 0
        self.fish.body.velocity = (0, 0)

        self.num_steps += 1
        self.board.erase()
        self.draw_shapes()

        right_eye_pos = (-np.cos(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
                         +np.sin(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])
        left_eye_pos = (+np.cos(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[0],
                        -np.sin(np.pi/2-self.fish.body.angle) * self.env_variables['eyes_biasx'] + self.fish.body.position[1])

        self.fish.left_eye.read(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
        self.fish.right_eye.read(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)

        in_light = self.fish.body.position[0] > self.dark_col

        if self.env_variables['hunger'] and self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.hungry, self.fish.stress]])
        elif self.env_variables['hunger']:
            internal_state = np.array([[in_light, self.fish.hungry]])
        elif self.env_variables['stress']:
            internal_state = np.array([[in_light, self.fish.stress]])
        else:
            internal_state = np.array([[in_light]])

        if save_frames or self.draw_screen:
            self.board.erase(bkg=self.env_variables['bkg_scatter'])
            self.draw_shapes()
            self.board.apply_light(self.dark_col, 0.7, 1)
            self.fish.left_eye.show_points(left_eye_pos[0], left_eye_pos[1], self.fish.body.angle)
            self.fish.right_eye.show_points(right_eye_pos[0], right_eye_pos[1], self.fish.body.angle)
            if save_frames:
                frame_buffer.append(self.output_frame(activations, internal_state, scale=0.25))
            if self.draw_screen:
                self.board_image.set_data(self.output_frame(activations, internal_state, scale=0.5) / 255.)
                plt.pause(0.000001)

        observation = np.dstack((self.fish.readings_to_photons(self.fish.left_eye.readings),
                                 self.fish.readings_to_photons(self.fish.right_eye.readings)))

        return observation, reward, internal_state, done, frame_buffer

    def create_stimuli(self, stimuli):
        for stimulus in stimuli:
            if "prey" in stimulus:
                self.create_prey()
            elif "predator" in stimulus:
                self.create_predator()

    @staticmethod
    def get_distance_for_size(stimulus, degree_size):
        if "prey" in stimulus:
            return 298.97 * np.exp(-0.133 * degree_size)
        elif "predator" in stimulus:
            return 298.97 * np.exp(-0.133 * degree_size/25)
        else:
            return 180

    def place_on_curve(self, stimulus_key, index, distance, angle):
        b = distance * np.sin(angle) + self.fish.body.position[0]
        a = distance * np.cos(angle) + self.fish.body.position[1]
        if "prey" in stimulus_key:
            self.prey_bodies[index].position = (a, b)
        elif "predator" in stimulus_key:
            self.predator_bodies[index].position = (a, b)

    def update_random_stimuli(self):
        # TODO: Add in baseline feature.
        stimuli_to_delete = []
        for i, stimulus, in enumerate(self.random_stimuli.keys()):
            if self.num_steps % self.unset_stimuli[stimulus]["interval"] == 0:
                if self.random_stimuli[stimulus]["steps"] > self.num_steps:
                    d = self.get_distance_for_size(stimulus, self.random_stimuli[stimulus]["size"])
                    theta = np.random.uniform(-0.75, 0.75) * np.pi
                    self.place_on_curve(stimulus, i, d, theta)
                else:
                    stimuli_to_delete.append(stimulus)
        for stimulus in stimuli_to_delete:
            del self.stimuli[stimulus]

    def get_new_angle(self, duration, current_steps):
        if self.moving_stimuli is False:
            if current_steps < 1:
                return 0.75 * np.pi
            else:
                progression = current_steps / duration
                return ((1.5 * progression) - 0.75) * np.pi
        else:
            if self.moving_stimuli is "Right":
                if current_steps < 1:
                    return 0.75 * np.pi
                else:
                    progression = current_steps / duration
                    return ((1.5 * progression) - 0.75) * np.pi
            else:
                if current_steps < 1:
                    return -0.75 * np.pi
                else:
                    progression = (duration - current_steps) / duration
                    return ((1.5 * progression) - 0.75) * np.pi

    @staticmethod
    def get_distance_moved_on_arc(theta_1, theta_2, d):
        # new_d = np.sqrt((p1[0]-p2[0])^2+(p1[1]-p2[1])^2)
        # angle = np.arccos((2*(d^2)-new_d^2)/(4*new_d))
        angle = abs(abs(theta_2)-abs(theta_1))
        return angle * d

    def get_new_distance(self, stimulus, interval, current_steps):
        if current_steps%interval == 0:
            current_steps = int(interval / 3)
        else:
            current_steps = (current_steps % interval) - (interval * 2/3)
        if "prey" in stimulus:
            sizes = np.linspace(5, 15, int(interval / 3) + 1)
        elif "predator" in stimulus:
            sizes = np.linspace(40, 80, int(interval/3)+1)
        else:
            print("Error")
            sizes = np.linspace(5, 15, int(interval / 3) + 1)

        if self.moving_stimuli == "Towards":
            progression = int(current_steps)
        elif self.moving_stimuli == "Away":
            progression = int(round((interval/3)-current_steps))
        else:
            print("Wrong motion parameter")
            progression = 0
        return self.get_distance_for_size(stimulus, sizes[progression])

    def update_unset_stimuli(self):
        # TODO: Still need to update so that can have multiple, sequential stimuli. Will require adding in onset into stimulus, as well as changing the baseline phase. Not useful for current requirements.
        stimuli_to_delete = []
        init_period = 1

        for stimulus in self.unset_stimuli.keys():
            i = int(stimulus.split()[1]) - 1
            if self.num_steps <= init_period:
                # Network initialisation period
                if "prey" in stimulus:
                    self.prey_bodies[i].position = (10, 10)
                elif "predator" in stimulus:
                    self.predator_bodies[i].position = (10, 10)
            else:

                if (self.num_steps-init_period) % self.unset_stimuli[stimulus]["interval"] == 0:
                    # Initialisation period
                    self.stimuli_information[stimulus]["Initialisation"] = self.num_steps
                    if "prey" in stimulus:
                        self.prey_bodies[i].position = (10, 10)
                    elif "predator" in stimulus:
                        self.predator_bodies[i].position = (10, 10)

                elif (self.num_steps-init_period) % self.unset_stimuli[stimulus]["interval"] == round(self.unset_stimuli[stimulus]["interval"]/3):
                    # Pre onset period
                    self.stimuli_information[stimulus]["Pre-onset"] = self.num_steps
                    if "prey" in stimulus:
                        self.prey_bodies[i].position = (10, 10)
                    elif "predator" in stimulus:
                        self.predator_bodies[i].position = (10, 10)

                elif (self.num_steps-init_period) % self.unset_stimuli[stimulus]["interval"] == round(2 * self.unset_stimuli[stimulus]["interval"]/3):
                    # Appearance period
                    if self.unset_stimuli[stimulus]["steps"]-init_period > (self.num_steps-init_period):
                        d = self.get_distance_for_size(stimulus, self.unset_stimuli[stimulus]["size"])
                        theta = self.get_new_angle(self.unset_stimuli[stimulus]["steps"]-init_period, (self.num_steps-init_period))
                        self.place_on_curve(stimulus, i, d, theta)
                        self.stimuli_information[stimulus]["Onset"] = self.num_steps
                        self.stimuli_information[stimulus]["Angle"] = theta
                        self.stimuli_information[stimulus]["Size"] = self.unset_stimuli[stimulus]["size"]

                        if self.moving_stimuli:
                            if self.moving_stimuli == "Left" or self.moving_stimuli == "Right":
                                self.stimuli_information[stimulus]["Direction"] = self.moving_stimuli
                                time = self.unset_stimuli[stimulus]["interval"]/3
                                theta2 = self.get_new_angle(self.unset_stimuli[stimulus]["steps"]-init_period, (self.num_steps-init_period) + self.unset_stimuli[stimulus]["interval"]/3)
                                d_moved = self.get_distance_moved_on_arc(theta, theta2, d)
                                self.stimuli_information[stimulus]["Velocity"] = d_moved/time
                            elif self.moving_stimuli == "Towards" or self.moving_stimuli == "Away":
                                self.stimuli_information[stimulus]["Direction"] = self.moving_stimuli
                                time = self.unset_stimuli[stimulus]["interval"] / 3
                                d2 = self.get_new_distance(stimulus, self.unset_stimuli[stimulus]["interval"], round((self.num_steps-init_period) + self.unset_stimuli[stimulus]["interval"]/3))
                                d_moved = abs(d2 - d)
                                self.stimuli_information[stimulus]["Velocity"] = d_moved/time
                            else:
                                print("Invalid *moving* parameter given")
                    else:
                        self.stimuli_information[stimulus]["Finish"] = self.num_steps
                        stimuli_to_delete.append(stimulus)

                else:
                    if self.moving_stimuli and self.unset_stimuli[stimulus]["interval"] * 2/3 < (self.num_steps-init_period) % self.unset_stimuli[stimulus]["interval"]:
                        if self.moving_stimuli == "Left" or self.moving_stimuli == "Right":
                            d = self.get_distance_for_size(stimulus, self.unset_stimuli[stimulus]["size"])
                            theta = self.get_new_angle(self.unset_stimuli[stimulus]["steps"]-init_period, (self.num_steps-init_period))
                            self.place_on_curve(stimulus, i, d, theta)
                        elif self.moving_stimuli == "Towards" or self.moving_stimuli == "Away":
                            d = self.get_new_distance(stimulus, self.unset_stimuli[stimulus]["interval"], (self.num_steps-init_period))
                            steps_for_angle = round(((self.num_steps-init_period)//self.unset_stimuli[stimulus]["interval"] * self.unset_stimuli[stimulus]["interval"]) + (2 * self.unset_stimuli[stimulus]["interval"] / 3))
                            theta = self.get_new_angle(self.unset_stimuli[stimulus]["steps"]-100, steps_for_angle)
                            self.place_on_curve(stimulus, i, d, theta)
                        else:
                            print("Invalid *moving* parameter given")

                    self.stimuli_information[stimulus] = {}
        for stimulus in stimuli_to_delete:
            del self.unset_stimuli[stimulus]

    def update_stimuli(self):
        """For use with set positioned stimuli."""
        finished_prey = []
        finished_predators = []
        for i, prey in enumerate(self.prey_positions):
            try:
                self.prey_bodies[i].position = (self.prey_positions[prey][self.num_steps][0],
                                                self.prey_positions[prey][self.num_steps][1])
            except IndexError:
                self.prey_bodies.pop(i)
                self.prey_shapes.pop(i)
                finished_prey.append(prey)

        for i, predator in enumerate(self.predator_positions):
            try:
                self.predator_bodies[i].position = (self.predator_positions[predator][self.num_steps][0],
                                                    self.predator_positions[predator][self.num_steps][1])
            except IndexError:
                self.predator_bodies.pop(i)
                self.predator_shapes.pop(i)
                finished_predators.append(predator)

        for item in finished_prey:
            del self.prey_positions[item]
        for item in finished_predators:
            del self.predator_positions[item]

    def create_positional_information(self, stimuli):
        for stimulus in stimuli:
            edge_index = 0
            if "prey" in stimulus:
                self.prey_positions[stimulus] = []
                while edge_index + 1 < len(stimuli[stimulus]):
                    positions = self.interpolate_stimuli_positions(stimuli[stimulus], edge_index)
                    self.prey_positions[stimulus] = self.prey_positions[stimulus] + positions
                    edge_index += 1
            elif "predator" in stimulus:
                self.predator_positions[stimulus] = []
                while edge_index + 1 < len(stimuli[stimulus]):
                    positions = self.interpolate_stimuli_positions(stimuli[stimulus], edge_index)
                    self.predator_positions[stimulus] = self.predator_positions[stimulus] + positions
                    edge_index += 1

    @staticmethod
    def interpolate_stimuli_positions(stimulus, edge_index):
        a = stimulus[edge_index]["position"]
        b = stimulus[edge_index + 1]["position"]
        t_interval = stimulus[edge_index + 1]["step"] - stimulus[edge_index]["step"]
        dx = (b[0] - a[0])/t_interval
        dy = (b[1] - a[1])/t_interval
        interpolated_positions = [[a[0]+dx*i, a[1]+dy*i] for i in range(t_interval)]
        return interpolated_positions
