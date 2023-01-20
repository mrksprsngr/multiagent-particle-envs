#!/usr/bin/env python
import gym
import argparse
import multiagent

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='"Simplespread-v2"',
                        help='Name of the environment that should be played.')
    args = parser.parse_args()

    env = gym.make(args.env)

    x = env.reset()
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        env.step(action)

