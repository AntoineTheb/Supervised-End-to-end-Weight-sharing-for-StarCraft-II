import sys
import numpy as np

argv = sys.argv[1:]

for state in np.load(argv[0])['states']:
    state.show()
    #state.toDataline().show()
