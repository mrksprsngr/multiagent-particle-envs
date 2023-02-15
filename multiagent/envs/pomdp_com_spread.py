from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.pomdp_com_spread import POMDPComSpreadScenario


class POMDPComSpreadEnv(MultiAgentEnv):
    def __init__(self):

        scenario = POMDPComSpreadScenario()
        world = scenario.make_world()

        MultiAgentEnv.__init__(
            self,
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
        )
