#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
from obstacleAviodanceDDPG import DDPG
from GazeboEnv import GazeboEnv
import numpy as np
import cv2


if __name__ == "__main__":

    MAX_EPISODES = 30000
    MAX_EP_STEPS = 50
    rospy.init_node('gazebo_world')
    env = GazeboEnv()
    var = 3
    rl = DDPG()
    loss = 0
    rl.load_model()

    for i in range(1, MAX_EPISODES):
        env.reset_world()
        print "*************ep={}***mc={}****************".format(i, rl.memory_counter)

        st = 0
        rw = 0

        observation = env.get_state()
        # 分成末端坐标和 rgbd

        while True:
            st += 1
            joints, view_state = observation
            action = rl.choose_action([joints, view_state])
            recent_end_goal, r, done = env.step(action)  # 执行一步
            observation_ = env.get_state()

            rl.store_transition(observation, action, r, observation_)

            if rl.memory_counter > 10000:  # and st % 2 == 0:

                var *= .9995
                print(".....................learn.....................")
                loss = rl.learn()
            rw += r
            print "r = {},  done = {}".format(r, done)
            print "reward = {}".format(rw)
            if done or st >= MAX_EP_STEPS:
                log = open('log.txt', 'a')
                log.write("ep={},reward={},loss={}\n".format(i, rw,loss))
                break

            observation = observation_
        print i
        if i % 10 == 0:
            print 'save models'
            rl.save_model()
