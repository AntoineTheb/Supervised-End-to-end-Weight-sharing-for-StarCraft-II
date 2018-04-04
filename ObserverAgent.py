#!/usr/bin/env python

from pysc2.lib import actions
from Data import State


class ObserverAgent():
    def __init__(self):
        self.states = []

    def getStates(self):
        return self.states

    def step(self, time_step, action):
        self.states.append(State(time_step, action))


class NoNoOp(ObserverAgent):
    def step(self, time_step, action):
        if action.function != actions.FUNCTIONS.no_op.id:
            super(NoNoOp, self).step(time_step, action)
