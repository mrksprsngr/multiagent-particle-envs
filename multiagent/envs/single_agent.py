import gym

from multiagent.environment import MultiAgentEnv
from multiagent.scenarios.single_agent import SingleAgentScenario


class SingleAgentEnv(gym.Wrapper):
    def __init__(self):
        scenario = SingleAgentScenario()
        world = scenario.make_world()

        env = MultiAgentEnv(
            world=world,
            reset_callback=scenario.reset_world,
            reward_callback=scenario.reward,
            observation_callback=scenario.observation,
            info_callback=None,
            done_callback=None,
            shared_viewer=True  # set to True to make rendering static
        )
        super(SingleAgentEnv, self).__init__(env)

        # unwrap lists since only one agent is in the scenario
        self.action_space = self.action_spaces[0]
        self.observation_space = self.observation_spaces[0]

    def step(self, action):
        xs, rs, dones, info = self.env.step([action])
        return xs[0], rs[0], dones[0], info

    def reset(self, **kwargs):
        xs = self.env.reset(**kwargs)
        return xs[0]

    def render(self, mode="human", **kwargs):
        return self.env.render(mode, **kwargs)