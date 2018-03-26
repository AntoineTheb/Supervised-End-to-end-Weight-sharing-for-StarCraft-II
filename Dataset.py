__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import numpy as np

from pysc2.lib import actions


class Dataset:
    def __init__(self):
        self.input_observations = []
        self.input_available_actions = []
        self.output_actions = []
        self.output_params = []
        self.weights = []

    def load(self, path):
        print("Loading data...")

        # TODO: load state length for each files
        valid_files = 0
        for f in os.listdir(path):
            if f.find(".npy") != -1:
                valid_files += 1

        file_index = -1
        for f in os.listdir(path):
            if f.find(".npy") != -1:
                file_index += 1
                if file_index % 100 == 0:
                    print("Loading file", file_index, "/", valid_files)
                file_name = f[:f.find(".npy")]
                states = np.load("{}/{}.npy".format(path, file_name))

                if len(self.input_observations) == 0:
                    self.input_observations = np.zeros((valid_files * len(states), 84, 84, 2))
                if len(self.input_available_actions) == 0:
                    self.input_available_actions = np.zeros((valid_files * len(states), 524))
                if len(self.output_actions) == 0:
                    self.output_actions = np.zeros((valid_files * len(states), 524))
                if len(self.output_params) == 0:
                    self.output_params = np.zeros((valid_files * len(states), 2))

                for i in range(0, len(states)):
                    state = states[i]

                    current_state = file_index * len(states) + i

                    self.input_observations[current_state] = state[0]

                    output_size = len(actions.FUNCTIONS)

                    available_actions = np.zeros(output_size)
                    for action_index in state[1]:
                        available_actions[action_index] = 1.0
                    self.input_available_actions[current_state] = available_actions

                    output_action = np.zeros(output_size)
                    output_action[state[2]] = 1.0
                    self.output_actions[current_state] = output_action

                    if np.shape(state[3]) == (2,):
                        image_size = np.shape(state[0])[0]
                        point = [float(state[3][1][0]) / image_size, float(state[3][1][1]) / image_size]
                    else:
                        point = [0, 0]
                    self.output_params[current_state] = np.array(point)

        assert len(self.input_observations) == len(self.input_available_actions) == len(self.output_actions) == len(self.output_params)

        self.weights = np.ones(self.output_actions.shape[0])
        # self.weights[self.output_actions[:, 7] == 1.] = 2000
        self.weights = [self.weights, np.ones(self.output_actions.shape[0])]

        print("input observations: ", np.shape(self.input_observations))
        print("input available actions ", np.shape(self.input_available_actions))
        print("output actions: ", np.shape(self.output_actions))
        print("output params: ", np.shape(self.output_params))
        print("weights: ", np.shape(self.weights))

        print(self.output_actions[42000])
        print(self.output_params[42000])
