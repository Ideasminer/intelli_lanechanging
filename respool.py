# !/usr/bin/env Python3
# -*- coding: utf-8 -*-

__author__ = 'Chen Jiayi'

'A module for computing and save simulation results'

# 1.Run simulation-communicate with sumo
# 2.Calculate results
# 2.1.Episode(default 0)
# 2.2.Travel time
# 2.3.State
# 2.4.TTC_leader
# 2.5.TTC_follower
# 2.6.TTC_agg
# 2.8.Average_speed
# 2.9.Reward
# 2.10.Lane-change times
# 3.Save results
# 3.1.Pool update
# 3.2.Exit and save

import pandas as pd
import traci
from traci import vehicle
import numpy as np
import random

class respool:
    def __init__(self):
        self.fp = 'respool.csv'             # save file path, default respool.csv
        self.cfg = 'freeway.sumocfg'        # cfg file path, default freeway.sumocfg
        self.agent = None
        self.period = 6
        self.log = 0                        # log for simulation time
        self.episode = [0]                  # episode, default [0]
        self.mode = 0                       # mode, default 0, represent simple simulation, 1 present DRL mode
        self.time = []                      # result parameters
        self.TTCl = []
        self.TTCf = []
        self.TTCll = []
        self.TTClf =[]
        self.TTCrl = []
        self.TTCrf = []
        self.avg_speed = []
        self.TTCa = []
        self.reward_list = []
        self.cl_num = 0
        self.cl_ratio = 1.0
        self.state = None
        self.reward = None
        self.lane_list = ['0','1']

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


        leader = vehicle.getLeader(self.agent, dist=1)
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
        return reward, TTC_agg, TTC_l, TTC_f, TTC_ll, TTC_lf, TTC_rl, TTC_rf, v_s

    def reset(self):
        try:
            traci.close()
        except:
            'closed'
        traci.start(['sumo', '-c', self.cfg])
        self.log = 0
        self.cl_num = 0
        self.log += self.period
        traci.simulationStep(step=self.log)
        self.agent = random.sample(vehicle.getIDList(), 1)[0]
        state = self.get_state()
        reward = self.get_reward()      # update state and reward
        return state

    def calculate(self):
        print(vehicle.getLaneChangeState(self.agent,1))

    def runner(self):
        self.reset()                        # first, start simulation and choose fake agent
        for time in range(100):
            last_lane = vehicle.getLaneIndex(self.agent)
            self.log += self.period             # update simulation log
            traci.simulationStep(self.log)      # set simulation step
            if self.agent not in vehicle.getIDList():
                self.time.append(traci.simulation.getTime())
                traci.close()
                break
            else:
                # self.calculate()
                now_lane = vehicle.getLaneIndex(self.agent)
                if now_lane != last_lane:
                    self.cl_num += 1
                    self.cl_ratio = 0.95
                else:
                    self.cl_ratio = 1
                self.get_state()                    # update state
                reward, TTC_agg, TTC_l, TTC_f, TTC_ll, TTC_lf, TTC_rl, TTC_rf, v_s = \
                self.get_reward()                   # update reward
                source = [reward, TTC_agg, TTC_l, TTC_f, TTC_ll, TTC_lf, TTC_rl, TTC_rf, v_s]
                target = [self.reward_list, self.TTCa, self.TTCl, self.TTCf,self.TTCll,\
                self.TTClf, self.TTCrl, self.TTCrf, self.avg_speed]
                [target[i].append(source[i]) for i in range(len(source))]

    def exit(self):
        df = self.episode.copy()
        df.extend([np.mean(i) for i in [self.reward_list, self.TTCa, self.TTCl, self.TTCf,self.TTCll,\
                self.TTClf, self.TTCrl, self.TTCrf, self.avg_speed]])
        df.extend(self.time)
        df.append(self.cl_num)
        # df = pd.DataFrame(pd.DataFrame(np.array(df)).values.T, columns=['episode', 'reward',
        #     'TTC_agg', 'TTC_l', 'TTC_f', 'TTC_ll', 'TTC_lf', 'TTC_rl', 'TTC_rf', 'v_s', 'time', 'cl_num'])
        return df

res = []
for i in range(100):
    temp = respool()
    temp.reset()
    temp.runner()
    res.append(temp.exit())

res = pd.DataFrame(pd.DataFrame(np.array(res)).values, columns=['episode', 'reward',
             'TTC_agg', 'TTC_l', 'TTC_f', 'TTC_ll', 'TTC_lf', 'TTC_rl', 'TTC_rf', 'v_s', 'time', 'cl_num'])
res.to_csv(r'respool.csv')
