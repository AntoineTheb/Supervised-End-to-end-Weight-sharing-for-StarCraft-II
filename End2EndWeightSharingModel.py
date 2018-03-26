__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np

from keras.layers import Dense, Conv2D, Input, Flatten, concatenate, Activation
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.models import model_from_json
from matplotlib import pyplot as plt

np.random.seed(1234)


class End2EndWeightSharingModel:
    def __init__(self):
        self.model = None

    def init_model(self, image_input_shape, actions_input_shape, output_size):
        visual_input = Input(shape=image_input_shape)
        contextual_input = Input(shape=actions_input_shape)

        image_model = Sequential()
        image_model.add(Conv2D(4, (3, 3), activation='relu', padding='same', input_shape=image_input_shape))
        image_model.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
        image_features = image_model(visual_input)

        attention_map = (Conv2D(1, (3, 3), padding='same'))(image_features)
        attention_map = Flatten()(attention_map)
        attention_map = Activation('softmax')(attention_map)

        encoded_image = Flatten()(image_features)
        encoded_image = Dense(512, activation='relu')(encoded_image)
        action_decoder = concatenate([encoded_image, contextual_input])
        action_decoder = Dense(1024, activation='relu')(action_decoder)
        action_decoder = Dense(1024, activation='relu')(action_decoder)
        action_decoder = Dense(output_size, activation='softmax')(action_decoder)

        self.model = Model(inputs=[visual_input, contextual_input], outputs=[action_decoder, attention_map])

        #optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        optimizer = Adam(lr=0.00001)
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=optimizer)

    def init_loaded_model(self):
        optimizer = Adam(lr=0.00001)
        self.model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy'], optimizer=optimizer)

    def fit(self, x_observations, x_available_actions, y_taken_actions, y_attention_positions, weights, epochs):
        return self.model.fit([x_observations, x_available_actions], [y_taken_actions, y_attention_positions],
                              shuffle=True, sample_weight=weights, validation_split=0.2,
                              epochs=epochs, batch_size=64, verbose=1)

    def predict(self, input_batch):
        pred = self.model.predict(input_batch, batch_size=1, verbose=0)
        action = np.argmax(pred[0][0])
        position = np.argmax(pred[1])
        plt.imshow(pred[1].reshape(84,84), cmap='gray')
        plt.show()
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
