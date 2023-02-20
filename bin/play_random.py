#!/usr/bin/env python
import gym
import argparse
import multiagent  # noqa

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='SimpleSpread-v0',
                        help='Name of the environment that should be played.')
    args = parser.parse_args()
    print(args.env)
    env = gym.make(args.env)

    while True:
        x = env.reset()
        for i in range(100):
            env.render()
            actions = []
            for acs in env.action_spaces:
                action = acs.sample()
                actions.append(action)
            env.step(actions)

