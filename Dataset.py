__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import numpy as np
import glob

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

        files = glob.glob("{}/*.npy".format(path))

        nbStates = 0
        for f in files:
            states = np.load(f)
            nbStates += len(states)

        self.input_observations = np.zeros((nbStates, 84, 84, 2))
        self.input_available_actions = np.zeros((nbStates, 524))
        self.output_actions = np.zeros((nbStates, 524))
        self.output_params = np.zeros((nbStates, 7056))

        offset = 0
        for f in files:
            print("Loading {}".format(f))
            for state in np.load(f):

                self.input_observations[offset] = state[0]

                output_size = len(actions.FUNCTIONS)

                available_actions = np.zeros(output_size)
                for action_index in state[1]:
                    available_actions[action_index] = 1.0
                self.input_available_actions[offset] = available_actions

                output_action = np.zeros(output_size)
                output_action[state[2]] = 1.0
                self.output_actions[offset] = output_action

                attention_map = np.zeros(state[0][:, :, 0].shape)
                if np.shape(state[3]) == (2,):
                    attention_map[tuple(state[3][1])] = 1
                self.output_params[offset] = attention_map.flatten()

                offset += 1

        assert len(self.input_observations) == len(self.input_available_actions) == len(self.output_actions) == len(self.output_params)

        self.weights = np.ones(self.output_actions.shape[0])
        self.weights[self.output_actions[:, 7] == 1.] = (self.output_actions[:, 7] == 0).sum() / (self.output_actions[:, 7] == 1).sum()
        self.weights = [self.weights, np.ones(self.output_actions.shape[0])]

        print("input observations: ", np.shape(self.input_observations))
        print("input available actions ", np.shape(self.input_available_actions))
        print("output actions: ", np.shape(self.output_actions))
        print("output params: ", np.shape(self.output_params))
        print("weights: ", np.shape(self.weights))
