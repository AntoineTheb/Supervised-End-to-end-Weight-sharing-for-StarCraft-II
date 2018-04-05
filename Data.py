__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np
import glob
import matplotlib.pyplot as plt

from pysc2.lib import features, actions


class Dataline:
    IMAGE_SHAPE = (84, 84)
    IMAGES_SHAPE = IMAGE_SHAPE + (2,)
    ACTION_SHAPE = (len(actions.FUNCTIONS),)
    PARAM_SHAPE = (np.prod(IMAGE_SHAPE),)

    def __init__(self):
        self.image = None
        self.available_actions = None
        self.action = None
        self.param = None

    def show(self):
        plt.figure(figsize=(8, 8))
        plt.subplot(2,2,1)
        plt.imshow(self.image[:,:,0])
        plt.title("player relative")

        plt.subplot(2,2,2)
        plt.imshow(self.image[:,:,1])
        plt.title("selected")

        plt.show()


class State:
    def __init__(self, observation, action=None):
        self.screen_player_relative = observation["screen"][features.SCREEN_FEATURES.player_relative.index]
        self.screen_selected = observation["screen"][features.SCREEN_FEATURES.selected.index]
        self.available_actions = observation["available_actions"]
        self.action = action

    def toDataline(self):
        dataline = Dataline()

        dataline.image = np.stack([self.screen_player_relative, self.screen_selected], axis=2)

        manyHotActions = np.zeros(Dataline.ACTION_SHAPE)
        for action_index in self.available_actions:
            manyHotActions[action_index] = 1.0
        dataline.available_actions = manyHotActions

        if self.action:
            oneHotAction = np.zeros(Dataline.ACTION_SHAPE)
            oneHotAction[self.action.function] = 1.0
            dataline.action = oneHotAction

            oneHotPosition = np.zeros(self.screen_player_relative.shape)
            if len(self.action.arguments) == 2:
                oneHotPosition[tuple(self.action.arguments[1])] = 1
            dataline.param = oneHotPosition.flatten()

        return dataline

    def show(self):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(self.screen_player_relative)
        plt.title("player relative")

        plt.subplot(2, 2, 2)
        plt.imshow(self.screen_selected)
        plt.title("selected")

        plt.subplot(2, 2, 3)
        plt.axis('off')
        plt.text(0, 0.5, "available actions\n{}\n\n\n\n{}".format(self.available_actions, self.action))

        plt.show()


class Dataset:
    def __init__(self):
        self.images = None
        self.available_actions = None
        self.actions = None
        self.params = None
        self.weights = None

    def load(self, path):
        print("Loading data...")

        files = glob.glob("{}/*.npz".format(path))

        nbStates = 0
        for f in files:
            states = np.load(f)['states']
            nbStates += len(states)

        self.images = np.zeros((nbStates,) + Dataline.IMAGES_SHAPE)
        self.available_actions = np.zeros((nbStates,) + Dataline.ACTION_SHAPE)
        self.actions = np.zeros((nbStates,) + Dataline.ACTION_SHAPE)
        self.params = np.zeros((nbStates,) + Dataline.PARAM_SHAPE)

        offset = 0
        for f in files:
            for state in np.load(f)['states']:
                if offset % 5000 == 0:
                    print("Loading state {} of {}".format(offset, nbStates))

                dataline = state.toDataline()
                self.images[offset] = dataline.image
                self.available_actions[offset] = dataline.available_actions
                self.actions[offset] = dataline.action
                self.params[offset] = dataline.param

                offset += 1

        assert len(self.images) == len(self.available_actions) == len(self.actions) == len(self.params)

        self.weights = np.ones(self.actions.shape[0])
        self.weights[self.actions[:, 7] == 1.] = (self.actions[:, 7] == 0).sum() / (self.actions[:, 7] == 1).sum()
        self.weights = [self.weights, np.ones(self.actions.shape[0])]

        print("input observations: ", np.shape(self.images))
        print("input available actions ", np.shape(self.available_actions))
        print("output actions: ", np.shape(self.actions))
        print("output params: ", np.shape(self.params))
        print("weights: ", np.shape(self.weights))
