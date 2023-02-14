from gym.envs.registration import register
from multiagent.environment import MultiAgentEnv

# Multiagent envs
# ----------------------------------------

register(
    id='Simple-v0',
    entry_point='multiagent.envs.simple:SimpleEnv',
    max_episode_steps=100,
)

register(
    id='SimpleSpread-v0',
    entry_point='multiagent.envs.simple_spread:SimpleSpreadEnv',
)
