import sys
import numpy as np
from collections import Counter
import glob

argv = sys.argv[1:]

states = np.load(argv[0])['states']
print("actions occurences: ", Counter([s.action.function for s in states]))
i = 0
for state in states:
    print("state:", i, "len(states)", len(states))
    i += 1
    state.toDataline().show()

# files = glob.glob("dataset_zerglings/*.npz")
# for f in files:
#     states = np.load(f)['states']
#     functions = [s.action.function for s in states]
#     count = Counter(functions)
#     if 7 in count:
#         firstSelectArmy = functions.index(7)
#         print("actions occurences: ", Counter([s.action.function for s in states]), "len(states): ", len(states), "first select army: ", firstSelectArmy)
#     else:
#         print("actions occurences: ", Counter([s.action.function for s in states]), "len(states): ", len(states))


