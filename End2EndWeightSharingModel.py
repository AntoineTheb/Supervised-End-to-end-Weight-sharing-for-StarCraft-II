__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import numpy as np

from keras.layers import Dense, Conv2D, Input, Flatten, concatenate, Activation, Lambda, Multiply
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.models import model_from_json
from matplotlib import pyplot as plt
import keras.backend as k
import keras.preprocessing.image as preprocessor

np.random.seed(1234)

from Data import Dataline


class End2EndWeightSharingModel:
    def __init__(self):
        self.model = None

    def init_model(self):
        visual_input = Input(shape=Dataline.IMAGES_SHAPE)
        contextual_input = Input(shape=Dataline.ACTION_SHAPE)

        image_model = Sequential()
        image_model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=Dataline.IMAGES_SHAPE))
        image_model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        image_features = image_model(visual_input)

        attention_map = (Conv2D(1, (3, 3), padding='same'))(image_features)
        attention_map = Flatten()(attention_map)
        attention_map = Activation('softmax')(attention_map)

        encoded_image = Flatten()(image_features)
        encoded_image = Dense(512, activation='relu')(encoded_image)

        action_decoder = concatenate([encoded_image, contextual_input])
        action_decoder = Dense(1024, activation='relu')(action_decoder)
        action_decoder = Dense(1024, activation='relu')(action_decoder)

        pi = Dense(Dataline.ACTION_SHAPE[0], activation='softmax')(action_decoder)
        pi_masked = Multiply()([pi, contextual_input])
        pi_normalized = Lambda(lambda x: x / k.sum(x, axis=1, keepdims=True), name='pi')(pi_masked)
        v = Dense(1, name='v')(action_decoder)

        self.model = Model(inputs=[visual_input, contextual_input], outputs=[pi_normalized, v, attention_map])

        #optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        optimizer = Adam(lr=0.00001)
        self.model.compile(loss=['categorical_crossentropy', 'mse', 'categorical_crossentropy'], optimizer=optimizer)

    def init_loaded_model(self):
        optimizer = Adam(lr=0.00001)
        self.model.compile(loss=['categorical_crossentropy', 'mse', 'categorical_crossentropy'], optimizer=optimizer)

    def fit(self, dataset, epochs, transformDataset=False):
        #tb_callback = callbacks.TensorBoard(log_dir="./logs_{}".format(name), histogram_freq=2, batch_size=64,
        #                                   write_graph=True, write_grads=False, write_images=True, embeddings_freq=0,
        #                                   embeddings_layer_names=None, embeddings_metadata=None)

        if(transformDataset):
            generator = self.genflow(dataset, True, True, True, True)
            self.model.fit_generator(generator, shuffle=True,
                                     epochs=epochs, verbose=1,
                                     # callbacks=[tb_callback],
                                     steps_per_epoch=len(dataset.images) / 64
                                     )
        else:
            self.model.fit([dataset.images, dataset.available_actions], [dataset.actions, dataset.values, dataset.params], shuffle=True,
                           epochs=epochs, sample_weight=dataset.weights, batch_size=64, verbose=1, #callbacks=[tb_callback],
                           validation_split=0.2)

    def predict(self, dataline):
        pred = self.model.predict([np.array([dataline.image]), np.array([dataline.available_actions])], batch_size=1, verbose=0)
        action = np.argmax(pred[0][0])
        position = np.argmax(pred[2])
        plt.imshow(pred[2].reshape(dataline.IMAGE_SHAPE), cmap='gray')
        plt.show()
        y, x = np.unravel_index(position, dataline.IMAGE_SHAPE)
        return action, [x, y]

    def genflow(self, dataset, centered, normalize, horizontalFlip, verticalFlip):
        datagen = preprocessor.ImageDataGenerator(
            featurewise_center=centered,
            featurewise_std_normalization=normalize,
            horizontal_flip=horizontalFlip,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            vertical_flip=verticalFlip)


        genImage = datagen.flow(dataset.images, dataset.params, batch_size=32, seed =7)
        genActionAvailable = datagen.flow(dataset.images, dataset.available_actions, batch_size=32, seed=7)
        genAction = datagen.flow(dataset.images, dataset.actions, batch_size=32, seed=7)

        while True:
            image = genImage.next()
            availaction = genActionAvailable.next()
            action = genAction.next()

            yield[image[0], availaction[1]], [action[1], image[1]]

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
