import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


colors = {
    u'm': (0.75, 0, 0.75),
    u'y': (0.75, 0.75, 0),
    u'c': (0.0, 0.75, 0.75),
    u'g': (0.0, 0.5, 0.0),
    u'r': (1.0, 0.0, 0.0),
    u'b': (0.0, 0.0, 1.0),
    u'k': (0.0, 0.0, 0.0),
}


class BlindSpreadScenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array(list(colors.values())[i])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array(list(colors.values())[i])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for ag, lm in zip(world.agents, world.landmarks):
            if ag is agent:
                rew -= np.linalg.norm(ag.state.p_pos - lm.state.p_pos)
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        agent_idx = -1
        # communication of all other agents
        comm = []
        other_pos = []
        for i, other in enumerate(world.agents):
            if other is agent: 
                agent_idx = i
                continue
            if not other.silent:
                comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        assert agent_idx >= 0, f"Did not find matching landmark for agent"
        # get positions of all entities in world frame
        landmark_pos = []
        for j, entity in enumerate(world.landmarks): 
            if j == agent_idx: 
                continue
            landmark_pos.append(entity.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + landmark_pos + other_pos + comm)
