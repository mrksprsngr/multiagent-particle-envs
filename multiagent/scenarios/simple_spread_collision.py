import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from multiagent_algs.utils.mpi_tools import mpi_print


class SimpleSpreadCollisionScenario(BaseScenario):
    def make_world(self): 
        # Initialize world and set properties (reset is done once at the end of this function)
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
            if i == 0:
                agent.color = np.array([0.35, 0.35, 0.85])
            else:     
                agent.color = np.array([0.85, 0.35, 0.35])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i == 0:
                landmark.color = np.array([0.30, 0.30, 0.80])
            else:
                landmark.color = np.array([0.80, 0.30, 0.30])
        # Randomly choose one diagonal
        positive_diagonal_index = np.random.randint(0, 2)
        negative_diagonal_index = np.random.randint(0, 2)
        # positive_diagonal_index = 0
        # negative_diagonal_index = 0
        # Set the position of the agents and landmarks
        agent_org_dist = 0.8
        landmark_org_dist = 0.4
        agent_pos = self.get_start_position(positive_diagonal_index, negative_diagonal_index, agent_org_dist, 'agent')
        landmark_pos = self.get_start_position(positive_diagonal_index, negative_diagonal_index, landmark_org_dist, 'landmark')
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = agent_pos[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = landmark_pos[i]
            landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        # markus: may change to reward based on specific landmark

        rew = 0
        for i, agent in enumerate(world.agents):
            landmark = world.landmarks[i]
            dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
            rew -= dist  # assuming rew is your reward variable
            if not all(-1 <= value <= 1 for value in agent.state.p_pos):
                rew -= 3

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
        other_pos_relative = []
        other_vel_relative = []
        other_pos_absolut = []
        other_vel_absolut = []  
        for other in world.agents:
            if other is agent: continue
            if not other.silent:
                comm.append(other.state.c)
            other_pos_relative.append(other.state.p_pos - agent.state.p_pos)
            other_vel_relative.append(other.state.p_vel - agent.state.p_vel)
            other_pos_absolut.append(other.state.p_pos)
            other_vel_absolut.append(other.state.p_vel)

        entity_pos = np.array(entity_pos)
        other_pos_relative = np.array(other_pos_relative)
        other_vel_relative = np.array(other_vel_relative)
        other_pos_absolut = np.array(other_pos_absolut)
        other_vel_absolut = np.array(other_vel_absolut)

        if len(comm) > 0:
            return np.concatenate([
                agent.state.p_vel.flatten(),
            ])
        else:
            return np.concatenate([
                agent.state.p_vel.flatten(),
                agent.state.p_pos.flatten(),
                entity_pos.flatten(),
                other_pos_relative.flatten(),
                other_vel_relative.flatten()
            ])

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        rew = self.reward(agent, world)
        collisions = self.collisions(agent, world)
        occupied_landmarks = self.occupied_landmarks(agent, world)
        min_dists = self.min_dists(agent, world)

        return (rew, collisions, min_dists, occupied_landmarks)

    def collisions(self, agent, world):
        collisions = 0
        if agent.collide:
            for a in world.agents:
                if a is not agent and self.is_collision(a, agent):
                    collisions += 1
        return collisions

    def occupied_landmarks(self, agent, world):
        occupied_landmarks = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            if min(dists) < 0.1:
                occupied_landmarks += 1
        return occupied_landmarks

    def min_dists(self, agent, world):
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
        return min_dists

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False


    def get_start_position(self, positive_diagonal_index, negative_diagonal_index, org_dist, entity):
        if entity == 'agent':
            pos = np.array([-org_dist * np.random.randn() * 0.001, -org_dist* np.random.randn() * 0.001]) if positive_diagonal_index == 0 else np.array([org_dist, org_dist])
            pos = np.vstack([pos, [-org_dist* np.random.randn() * 0.001, org_dist*np.random.randn() *0.001] if negative_diagonal_index == 0 else [org_dist, -org_dist]])
        elif entity == 'landmark':
            # Switching the diagonal index for landmark 
            pos = np.array([org_dist, org_dist]) if positive_diagonal_index == 0 else np.array([-org_dist, -org_dist])
            pos = np.vstack([pos, [org_dist, -org_dist] if negative_diagonal_index == 0 else [-org_dist, org_dist]])
        return pos