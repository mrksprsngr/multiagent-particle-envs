#!/usr/bin/env python
import gym
import multiagent  # noqa

if __name__ == '__main__':
    env = gym.make("Simple-v0")

    for i in range(10):
        # start a new eposide
        x = env.reset()

        N = 100
        for _ in range(100):
            # run N steps for each episode
            env.render()
            action = env.action_space.sample()
            print(action)
            env.step(action)
    env.close()

