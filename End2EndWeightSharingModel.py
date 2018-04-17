__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np

from keras.layers import Conv2D, Input, Flatten, Activation, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from keras import callbacks

np.random.seed(1234)

from Data import Dataline


class End2EndWeightSharingModel:
    def __init__(self):
        self.model = None

    def init_model(self):
        visual_input = Input(shape=Dataline.IMAGES_SHAPE)

        image_features = Conv2D(4, (3, 3), activation='relu', padding='same')(visual_input)
        image_features = Conv2D(4, (3, 3), activation='relu', padding='same')(image_features)

        attention_map = Conv2D(3, (3, 3), activation='relu', padding='same')(image_features)
        attention_map = Flatten()(attention_map)
        attention_map = Activation('softmax')(attention_map)

        self.model = Model(inputs=[visual_input], outputs=[attention_map])

        optimizer = Adam(lr=0.001)
        self.model.compile(loss=['categorical_crossentropy'], optimizer=optimizer)

    def init_loaded_model(self):
        optimizer = Adam(lr=0.001)
        self.model.compile(loss=['categorical_crossentropy'], optimizer=optimizer)

    def fit(self, dataset, epochs):
        #tb_callback = callbacks.TensorBoard(log_dir="./logs_{}".format(name), histogram_freq=2, batch_size=64,
        #                                   write_graph=True, write_grads=False, write_images=True, embeddings_freq=0,
        #                                   embeddings_layer_names=None, embeddings_metadata=None)
        self.model.fit(dataset.images, dataset.params.reshape(dataset.params.shape[0], -1), shuffle=True,
                       epochs=epochs, sample_weight=dataset.weights, batch_size=64, verbose=1, #callbacks=[tb_callback],
                       validation_split=0.2)

    def predict(self, dataline):
        pred = self.model.predict(np.expand_dims(dataline.image, 0), batch_size=1, verbose=0)
        position_action = np.unravel_index(np.argmax(pred), Dataline.PARAM_SHAPE)
        position = position_action[0:2][::-1]
        action = Dataline.indexToAction[position_action[2]]

        End2EndWeightSharingModel.show_prediction(pred)

        return action, position

    def save(self, name):
        model_json = self.model.to_json()
        with open("bin/{}.json".format(name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("bin/{}.h5".format(name))

    def load(self, name):
        with open("bin/{}.json".format(name), "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("bin/{}.h5".format(name))

    @staticmethod
    def show_prediction(pred):
        pred = pred.reshape(Dataline.PARAM_SHAPE)
        plt.figure(figsize=(8, 8))

        plt.subplot(2,2,1)
        plt.imshow(pred[:,:,0], cmap=cm.gray, vmin=0, vmax=1)
        plt.title("select point")

        plt.subplot(2,2,2)
        plt.imshow(pred[:,:,1], cmap=cm.gray, vmin=0, vmax=1)
        plt.title("attack screen")

        plt.subplot(2,2,3)
        plt.imshow(pred[:,:,2], cmap=cm.gray, vmin=0, vmax=1)
        plt.title("move screen")

        plt.show()
