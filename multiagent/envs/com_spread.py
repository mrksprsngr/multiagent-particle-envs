from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.com_spread import ComSpreadScenario


class ComSpreadEnv(MultiAgentEnv):
    def __init__(self):

        scenario = ComSpreadScenario()
        world = scenario.make_world()

        MultiAgentEnv.__init__(
            self,
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
        )
