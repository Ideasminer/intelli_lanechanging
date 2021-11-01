# !/usr/bin/env Python3
# -*- coding: utf-8 -*-

__author__ = 'Chen Jiayi'

# import packages


from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import traci
from traci import vehicle, lane, lanearea
import random
import matplotlib.pyplot as plt


# 1.Establish Simulation class
# 1.1.Have basic parameters like action_size and state_size
# 1.2.Can take actions and get array
# 1.2.1.Actions have 3 dimensions, which are keep/left/right
# 1.2.1.1.Keep:index 0, if keep, no action taken actually
# 1.2.1.2.Left:index 1, if left, change to left lane in period
# 1.2.1.3.Right:index 2, if right, change to right lane in period
# 1.2.2.Return array have at least 3 dimensions, which are next_state, reward and done
# 2.Establish DQNAgent
# 3.Combination


# Global Parameters


EPISODES = 5000

class Curve:
    def __init__(self,fp):
        self.reward_pool = []
        self.episode_pool = []
        self.fp = fp

    def save(self,reward,episode):
        self.reward_pool.append(reward)
        self.episode_pool.append(episode)

    def visualize(self):
        plt.plot(self.episode_pool,self.reward_pool)
        plt.savefig(self.fp)
        plt.close()

class simulation:
    def __init__(self):
        self.agent = None
        self.period = 6
        self.action_size = 3
        self.state_size = 28          # Now only consider self v leader v distance
        self.log = 0
        self.state = [50, 50, 1000]
        self.reward = 100
        self.lane_list = ['0','1']
        self.cl_ratio = 1.0
        self.cl_num = 0

    def get_state(self):
        # self length, lane index, v, lane position
        # self leader length, lane index, v, lane position
        # self follower length, lane index, v, lane position
        # left leader length, lane index, v, lane position
        # left follower length, lane index, v, lane position
        # right leader length, lane index, v, lane position
        # right follower length, lane index, v, lane position
        # 28 dimensions


        if self.agent == None:
            return None
        # self parameters
        length_s = vehicle.getLength(self.agent)
        lane_s = vehicle.getLaneIndex(self.agent)
        v_s = vehicle.getSpeed(self.agent)
        position_s = vehicle.getLanePosition(self.agent)


        # self leader parameters


        leader = vehicle.getLeader(self.agent, dist=100)
        if leader == None:
            length_l, lane_l, v_l, position_l = 5.0, lane_s, 30, position_s + 1000
        else:
            l = leader[0]
            length_l = vehicle.getLength(l)
            lane_l = vehicle.getLaneIndex(l)
            v_l = vehicle.getSpeed(l)
            position_l = vehicle.getLanePosition(l)


        # self follower parameters


        sf_list = [i for i in vehicle.getIDList() if vehicle.getLaneIndex(i) == vehicle.getLaneIndex(self.agent)]
        sf_p_list = [vehicle.getLanePosition(self.agent) - vehicle.getLanePosition(j) for j in sf_list
        if vehicle.getLanePosition(j) < vehicle.getLanePosition(self.agent)]
        if sf_p_list != []:
            follower = sf_list[sf_p_list.index(min(sf_p_list))]
            length_f = vehicle.getLength(follower)
            lane_f = vehicle.getLaneIndex(follower)
            v_f = vehicle.getSpeed(follower)
            position_f = vehicle.getLanePosition(follower)
        else:
            length_f, lane_f, v_f, position_f = 5.0, lane_s, 30, position_s - 1000


        # left leader parameters


        left_l = vehicle.getLeftLeaders(self.agent)
        if left_l != []:
            length_ll = vehicle.getLength(left_l[0][0])
            lane_ll = vehicle.getLaneIndex(left_l[0][0])
            v_ll = vehicle.getSpeed(left_l[0][0])
            position_ll = vehicle.getLanePosition(left_l[0][0])
        else:
            length_ll, lane_ll, v_ll, position_ll = 5.0, min(0, lane_s - 1), 30, position_s + 1000


        # left follower parameters


        left_f = vehicle.getLeftFollowers(self.agent)
        if left_f != []:
            length_lf = vehicle.getLength(left_f[0][0])
            lane_lf = vehicle.getLaneIndex(left_f[0][0])
            v_lf = vehicle.getSpeed(left_f[0][0])
            position_lf = vehicle.getLanePosition(left_f[0][0])
        else:
            length_lf, lane_lf, v_lf, position_lf = 5.0, min(0, lane_s - 1), 30, position_s - 1000


        # right leader parameters


        right_l = vehicle.getRightLeaders(self.agent)
        if right_l != []:
            length_rl = vehicle.getLength(right_l[0][0])
            lane_rl = vehicle.getLaneIndex(right_l[0][0])
            v_rl = vehicle.getSpeed(right_l[0][0])
            position_rl = vehicle.getLanePosition(right_l[0][0])
        else:
            length_rl, lane_rl, v_rl, position_rl = 5.0, max(lane_s + 1, len(self.lane_list)), 30, position_s + 1000


        # right follower parameters


        right_f = vehicle.getRightFollowers(self.agent)
        if right_f != []:
            length_rf = vehicle.getLength(right_f[0][0])
            lane_rf = vehicle.getLaneIndex(right_f[0][0])
            v_rf = vehicle.getSpeed(right_f[0][0])
            position_rf = vehicle.getLanePosition(right_f[0][0])
        else:
            length_rf, lane_rf, v_rf, position_rf = 5.0, max(lane_s + 1, len(self.lane_list)), 30, position_s - 1000


        state = [length_s, lane_s, v_s, position_s, length_l, lane_l, v_l, position_l,
        length_f, lane_f, v_f, position_f, length_ll, lane_ll, v_ll, position_ll,
        length_lf, lane_lf, v_lf, position_lf, length_rl, lane_rl, v_rl, position_rl,
        length_rf, lane_rf, v_rf, position_rf]

        self.state = state
        return state

    def get_reward(self):
        # (v / 30) * TTC_agg * cl_ratio
        # TTC_agg = 0.3 * TTC_l + 0.2* TTC_f + 0.15 * TTC_ll + 0.1 * TTC_lf + 0.15 * TTC_rl + 0.1 * TTC_rf
        # TTC: (distance - length_front / diff(v))
        # cl_ratio: change lane ratio, if lane changed, ratio = 0.8 else 1.0


        if self.state == None:
            return None

        length_s = lane_s = v_s = position_s = length_l = lane_l = v_l = position_l = \
        length_f = lane_f = v_f = position_f = length_ll = lane_ll = v_ll = position_ll = \
        length_lf = lane_lf = v_lf = position_lf = length_rl = lane_rl = v_rl = position_rl = \
        length_rf = lane_rf = v_rf = position_rf = 0         # initialize variables


        length_s, lane_s, v_s, position_s, length_l, lane_l, v_l, position_l,\
        length_f, lane_f, v_f, position_f, length_ll, lane_ll, v_ll, position_ll,\
        length_lf, lane_lf, v_lf, position_lf, length_rl, lane_rl, v_rl, position_rl,\
        length_rf, lane_rf, v_rf, position_rf = self.state


        TTC_l = (position_l - position_s - length_l) / (v_s - v_l) if v_s > v_l else 1000
        TTC_f = (position_s - position_f - length_s) / (v_f - v_s) if v_f > v_s else 1000
        TTC_ll = (position_ll - position_s - length_ll) / (v_s - v_ll) if v_s > v_ll else 1000
        TTC_lf = (position_s - position_lf - length_s) / (v_lf - v_s) if v_lf > v_s else 1000
        TTC_rl = (position_rl - position_s - length_rl) / (v_s - v_rl) if v_s > v_rl else 1000
        TTC_rf = (position_s - position_rf - length_s) / (v_rf - v_s) if v_rf > v_s else 1000
        TTC_agg = 0.3 * TTC_l + 0.2* TTC_f + 0.15 * TTC_ll + 0.1 * TTC_lf + 0.15 * TTC_rl + 0.1 * TTC_rf
        v_ratio = v_s / 30


        reward =v_ratio * TTC_agg * self.cl_ratio
        self.reward = reward
        return reward

    def reset(self):
        try:
            traci.close()
        except:
            'closed'
        traci.start(['sumo', '-c', 'freeway.sumocfg'])
        self.log = 0
        self.cl_num = 0
        self.log += self.period
        traci.simulationStep(step=self.log)
        self.agent = random.sample(vehicle.getIDList(), 1)[0]
        state = self.get_state()
        reward = self.get_reward()      # update state and reward
        return state

    def step(self, action):
        # frist recognize contemporary lane
        # then decide the lane agent will change to
        # step lane-changing behavior
        # update state
        done = False
        lane_now = vehicle.getLaneIndex(self.agent)
        if action == 1 and lane_now - 1 >= 0:
            vehicle.changeLane(self.agent, lane_now - 1, self.period/2)
            self.cl_ratio = 0.95
            self.cl_num += 1
        elif action == 2 and (lane_now + 1) <= (len(self.lane_list)-1):
            vehicle.changeLane(self.agent, lane_now + 1, self.period/2)
            self.cl_ratio = 0.95
            self.cl_num += 1
        self.log += self.period
        traci.simulationStep(self.log)
        if self.agent not in vehicle.getIDList():
            done = True
            state = self.state      # now state is not updated, former state
            reward = 1000
            traci.close()
        else:
            done = False
            self.get_reward()
            self.get_state()
            state = self.state
            reward = self.reward
        return state, reward, done

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95                   # discount parameter
        self.epsilon = 1.0                  # exploration rate
        self.epsilon_min = 0.01             # final epsilon
        self.epsilon_decay = 0.995          # decay rate
        self.learning_rate = 0.001          # learning rate
        self.model = self._build_model()    # neural network
        self.avg_q = 0

    def _build_model(self):
        # Neural Network for Q target(build with keras)
        model = Sequential()
        # input layer
        model.add(Dense(5, input_dim=self.state_size, activation='relu'))
        # hidden layer
        model.add(Dense(5, activation='relu'))
        # full connection
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)   # greedy step
        act_values = self.model.predict(state)          # predict action
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        q_list = []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                    np.amax(self.model.predict(state)[0]))
            target_f = self.model.predict(state)
            # target_f is the prediction of actions
            # if action_size equals n
            # target_f[0] have n dimensions
            # each dimension represent corresponding Q value
            # then give target_f[0][action],
            # which is the best action with calculated true Q value for training
            target_f[0][action] = target
            q_list.append(target)
            self.model.fit(state, target_f, epochs=1, verbose=0)
        self.avg_q = sum(q_list) / len(q_list)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)   # for transfer learning

    def save(self, name):
        self.model.save_weights(name)   # save nn's weight

def train(env, EPISODES):
    state_size = env.state_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32
    avg_q_list = []
    avg_cl_list = []
    q_curve = Curve(r'avgQ.png')
    cl_curve = Curve(r'avgCl.png')


    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)           # index of the best action
            print(action)
            next_state, reward, done = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, avg_q: {}, lane-change: {}, e: {:.2}"
                      .format(e, EPISODES, agent.avg_q, env.cl_num, agent.epsilon))
                avg_q_list.append(agent.avg_q)
                avg_cl_list.append(env.cl_num)
                if e % 50 == 0:
                    q_curve.save(sum(avg_q_list[-50:-1]) / 50, e)
                    q_curve.visualize()
                    cl_curve.save(sum(avg_cl_list[-50:-1]) / 50, e)
                    cl_curve.visualize()
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)


# TEST

if __name__ == '__main__':
    env = simulation()
    train(env, EPISODES)
