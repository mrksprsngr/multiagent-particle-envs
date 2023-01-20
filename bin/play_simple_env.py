#!/usr/bin/env python
import gym
import multiagent  # noqa

if __name__ == '__main__':
    env = gym.make("Simple-v0")

    x = env.reset()
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        env.step(action)
    env.close()

