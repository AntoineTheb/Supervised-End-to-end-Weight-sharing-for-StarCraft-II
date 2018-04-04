#!/usr/bin/env python
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import sys

from End2EndWeightSharingModel import *
from Data import Dataset

import os.path

np.random.seed(1234)

argv = sys.argv[1:]

if len(argv) < 2:
    print("Error: not enough argument supplied:")
    print("train.py <name> <nb_epochs>")
    exit(0)
else:
    name = argv[0]
    epochs = int(argv[1])

dataset = Dataset()
dataset.load("dataset_{}".format(name))

model = End2EndWeightSharingModel()

if os.path.isfile("bin/agent_{}.h5".format(name)) and os.path.isfile("bin/agent_{}.json".format(name)):
    model.load("agent_{}".format(name))
    model.init_loaded_model()
else:
    model.init_model()

training_results = model.fit(dataset, epochs)
# print(training_results.history)
model.save("agent_{}".format(name))
