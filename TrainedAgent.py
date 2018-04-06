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

        if action in obs.observation["available_actions"]:
            # print("action is available: ", action, position)
            if action == 7:
                print("select army")
        else:
            # if action not in obs.observation["available_actions"]:
            print("action", action, "is not available")
            action = np.random.choice(obs.observation["available_actions"])
            print("action", action, "was produced")

        if action == actions.FUNCTIONS.no_op.id:
            params = []
        elif action == actions.FUNCTIONS.Move_screen.id:
            params = [[0], position]
        elif action == actions.FUNCTIONS.select_army.id:
            params = [[0]]
        elif action == actions.FUNCTIONS.Attack_screen.id:
            params = [[0], position]
        else:
            params = [[np.random.randint(0, size) for size in arg.sizes] for arg in
                      self.action_spec.functions[action].args]

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