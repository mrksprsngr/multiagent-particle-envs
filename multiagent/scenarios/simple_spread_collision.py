import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class SimpleSpreadCollisionScenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = 2
        num_landmarks = 2
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
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # Randomly choose one diagonal
        diagonal_index = np.random.randint(0, 2)
        diagonal_agent = np.array([[0.8, 0.8], [-0.8, -0.8]]) if diagonal_index == 0 else np.array([[0.8, -0.8], [-0.8, 0.8]])
        diagonal_landmark = np.array([[0.8, 0.8], [-0.8, -0.8]]) if diagonal_index == 1 else np.array([[0.8, -0.8], [-0.8, 0.8]])
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = diagonal_agent[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = diagonal_landmark[i]
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
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        
        for other in world.agents:
            if other is agent: continue
            if not other.silent:
                comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        entity_pos = np.array(entity_pos)
        other_pos = np.array(other_pos)
        if len(comm) > 0:
            return np.concatenate([
                agent.state.p_vel.flatten(),
                agent.state.p_pos.flatten(),
                entity_pos.flatten(),
                other_pos.flatten(),
                np.asarray(comm)
            ])
        else:
            return np.concatenate([
                agent.state.p_vel.flatten(),
                agent.state.p_pos.flatten(),
                entity_pos.flatten(),
                other_pos.flatten()
            ])
