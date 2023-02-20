from gym.envs.registration import register
from multiagent.environment import MultiAgentEnv

# Multiagent envs
# ----------------------------------------

register(
    id='Simple-v0',
    entry_point='multiagent.envs.simple:SimpleEnv',
    max_episode_steps=100,
)

# Spread envs
# ----------------------------------------
register(
    id='SimpleSpread-v0',
    entry_point='multiagent.envs.simple_spread:SimpleSpreadEnv',
)

register(
    id='ComSpread-v0',
    entry_point='multiagent.envs.com_spread:ComSpreadEnv',
)

register(
    id='POMDPComSpreadEnv-v0',
    entry_point='multiagent.envs.pomdp_com_spread:POMDPComSpreadEnv',
)

register(
    id='BlindSpread-v0',
    entry_point='multiagent.envs.blind_spread:BlindSpreadEnv',
)