__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from pysc2.agents import base_agent
from pysc2.lib import actions

from End2EndWeightSharingModel import *
from Data import State

np.random.seed(1234)


class TrainedAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(TrainedAgent, self).step(obs)

        action, position = self.model.predict(State(obs.observation).toDataline())

        if action in Dataline.actionToIndex and action in obs.observation["available_actions"]:
            print("took available action:", actions.FUNCTIONS[action].name)
            params = [[0], position]
        else:
            print("took not available action:", actions.FUNCTIONS[action].name, "; changed for no_op!")
            action = actions.FUNCTIONS.no_op.id
            params = []

        return actions.FunctionCall(action, params)


class AgentRoaches(TrainedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.model = End2EndWeightSharingModel()
        self.model.load("agent_roaches")


class AgentBeacon(TrainedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.model = End2EndWeightSharingModel()
        self.model.load("agent_beacon")


class AgentMineral(TrainedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.model = End2EndWeightSharingModel()
        self.model.load("agent_mineral")


class AgentMinerals(TrainedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.model = End2EndWeightSharingModel()
        self.model.load("agent_minerals")

class AgentZerglings(TrainedAgent):
    def __init__(self):
        base_agent.BaseAgent.__init__(self)
        self.model = End2EndWeightSharingModel()
        self.model.load("agent_zerglings")