__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.utils import to_categorical

from pysc2.lib import features, actions


class Dataline:
    IMAGE_SHAPE = (84, 84)
    IMAGES_SHAPE = IMAGE_SHAPE + (5,)
    PARAM_SHAPE = IMAGE_SHAPE + (3,)
    actionToIndex = {2:0, 12:1, 331:2}  # select point, attack screen, move screen
    indexToAction = {0:2, 1:12, 2:331}

    def __init__(self):
        self.image = None
        self.param = None
        self.weight = None

    def show(self):
        plt.figure(figsize=(8, 8))

        plt.subplot(3,3,1)
        plt.imshow(self.image[:,:,0])
        plt.title("unit type - banelings")

        plt.subplot(3,3,2)
        plt.imshow(self.image[:,:,1])
        plt.title("unit type - marines")

        plt.subplot(3,3,3)
        plt.imshow(self.image[:,:,2])
        plt.title("unit type - zerglings")

        plt.subplot(3,3,4)
        plt.imshow(self.image[:,:,3])
        plt.title("selected")

        plt.subplot(3,3,5)
        plt.imshow(self.image[:,:,4])
        plt.title("unit_hit_point_ratio")

        if self.param is not None:
            plt.subplot(3,3,6)
            plt.imshow(self.param[:,:,0], cmap=cm.gray, vmin=0, vmax=1)
            plt.title("select point")

            plt.subplot(3,3,7)
            plt.imshow(self.param[:,:,1], cmap=cm.gray, vmin=0, vmax=1)
            plt.title("attack screen")

            plt.subplot(3,3,8)
            plt.imshow(self.param[:,:,2], cmap=cm.gray, vmin=0, vmax=1)
            plt.title("move screen")

        plt.show()


class State:
    def __init__(self, observation, action=None):
        self.screen_unit_type = observation["screen"][features.SCREEN_FEATURES.unit_type.index]
        self.screen_unit_hit_point_ratio = observation["screen"][features.SCREEN_FEATURES.unit_hit_points_ratio.index]
        self.screen_selected = observation["screen"][features.SCREEN_FEATURES.selected.index]
        self.available_actions = observation["available_actions"]
        self.action = action

    def toDataline(self):
        dataline = Dataline()

        unit_type = self.screen_unit_type
        unit_type[unit_type == 9] = 1
        unit_type[unit_type == 48] = 2
        unit_type[unit_type == 105] = 3
        dataline.image = np.concatenate((to_categorical(unit_type, 4)[:,:,1:], # Not background
                                         np.expand_dims(self.screen_selected, axis=2),
                                         np.expand_dims(self.screen_unit_hit_point_ratio, axis=2)),
                                        axis=2)

        if self.action:
            one_hot_position = np.zeros(Dataline.PARAM_SHAPE)
            if self.action.function in [2, 12, 331]:
                one_hot_position[tuple(self.action.arguments[1])[::-1] + (Dataline.actionToIndex[self.action.function],)] = 1
                dataline.weight = 1
            else:
                dataline.weight = 0
            dataline.param = one_hot_position

        return dataline


class Dataset:
    def __init__(self):
        self.images = None
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
        self.params = np.zeros((nbStates,) + Dataline.PARAM_SHAPE)
        self.weights = np.ones(nbStates)

        offset = 0
        for f in files:
            for state in np.load(f)['states']:
                if offset % 5000 == 0:
                    print("Loading state {} of {}".format(offset, nbStates))

                dataline = state.toDataline()
                self.images[offset] = dataline.image
                self.params[offset] = dataline.param
                self.weights[offset] = dataline.weight

                offset += 1

        if np.isnan(self.images).any():
            print("nan found in images", np.argwhere(np.isnan(self.images)))

        if np.isnan(self.params).any():
            print("nan found in params", np.argwhere(np.isnan(self.params)))

        if np.isnan(self.weights).any():
            print("nan found in weight", np.argwhere(np.isnan(self.weights)))

        print("input observations: ", np.shape(self.images))
        print("output params: ", np.shape(self.params))
        print("weights: ", np.shape(self.weights))
        print(self.weights.sum(), "valid on", nbStates)
