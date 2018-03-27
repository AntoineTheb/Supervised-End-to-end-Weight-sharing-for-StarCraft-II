__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np

from keras.layers import Dense, Conv2D, Input, Flatten, concatenate, Dropout
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.models import model_from_json
from keras import regularizers
from keras import callbacks

np.random.seed(1234)


class End2EndWeightSharingModel:
    def __init__(self):
        self.model = None

    def init_model(self, image_input_shape, actions_input_shape, output_size):
        # regularizer = regularizers.l2(0.001)
        regularizer = None
        use_dropout = False

        image_model = Sequential()

        image_model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=image_input_shape, padding='same', kernel_regularizer=regularizer))
        if use_dropout: image_model.add(Dropout(0.1))
        image_model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu', padding='same', kernel_regularizer=regularizer))
        if use_dropout: image_model.add(Dropout(0.4))
        image_model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizer))
        if use_dropout: image_model.add(Dropout(0.4))
        image_model.add(Flatten())
        image_model.add(Dense(512, activation='relu', kernel_regularizer=regularizer))
        if use_dropout: image_model.add(Dropout(0.4))

        visual_input = Input(shape=image_input_shape)
        encoded_image = image_model(visual_input)

        contextual_input = Input(shape=actions_input_shape)

        action_decoder = concatenate([encoded_image, contextual_input])
        action_decoder = Dense(1024, activation='relu', kernel_regularizer=regularizer)(action_decoder)
        if use_dropout: action_decoder = Dropout(0.1)(action_decoder)
        action_decoder = Dense(1024, activation='relu', kernel_regularizer=regularizer)(action_decoder)
        if use_dropout: action_decoder = Dropout(0.4)(action_decoder)
        action_decoder = Dense(output_size, activation='softmax', kernel_regularizer=regularizer)(action_decoder)

        regressor = Dense(512, activation='relu', kernel_regularizer=regularizer)(encoded_image)
        if use_dropout: regressor = Dropout(0.4)(regressor)
        regressor = Dense(128, activation='relu', kernel_regularizer=regularizer)(regressor)
        if use_dropout: regressor = Dropout(0.4)(regressor)
        regressor = Dense(2, activation='sigmoid', kernel_regularizer=regularizer)(regressor)

        self.model = Model(inputs=[visual_input, contextual_input], outputs=[action_decoder, regressor])

        #optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        optimizer = Adam(lr=0.00001)
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], #loss_weights=[14000, 1],
                           optimizer=optimizer)

    def init_loaded_model(self):
        optimizer = Adam(lr=0.00001)
        self.model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], #loss_weights=[14000, 1],
                           optimizer=optimizer)

    def fit(self, x_observations, x_available_actions, y_taken_actions, y_attention_positions, weights, epochs, name):
        tb_callback = callbacks.TensorBoard(log_dir="./logs_{}".format(name), histogram_freq=2, batch_size=64,
                                            write_graph=True, write_grads=False, write_images=True, embeddings_freq=0,
                                            embeddings_layer_names=None, embeddings_metadata=None)
        self.model.fit([x_observations, x_available_actions], [y_taken_actions, y_attention_positions], shuffle=True,
                       epochs=epochs, sample_weight=weights, batch_size=64, verbose=1, # callbacks=[tb_callback],
                       validation_split=0.2)

    def predict(self, input_batch):
        pred = self.model.predict(input_batch, batch_size=1, verbose=0)
        action = np.argmax(pred[0][0])
        position = pred[1][0]

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
