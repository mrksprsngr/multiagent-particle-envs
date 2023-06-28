from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.simple_spread_collision import SimpleSpreadCollisionScenario


class SimpleSpreadCollisionEnv(MultiAgentEnv):
    def __init__(self):

        scenario = SimpleSpreadCollisionScenario()
        world = scenario.make_world()

        MultiAgentEnv.__init__(
            self,
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            info_callback=None,
            done_callback=None,
            shared_viewer=True
        )
