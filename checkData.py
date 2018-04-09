import sys
import numpy as np
from collections import Counter

argv = sys.argv[1:]

states = np.load(argv[0])['states']
print("actions occurences: ", Counter([s.action.function for s in states]))
for state in states:
    #state.show()
    state.toDataline().show()
