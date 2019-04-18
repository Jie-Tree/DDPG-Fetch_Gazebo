#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
import torch
import time
from obstacleAviodanceDDPG import DDPG
from GazeboEnv import GazeboEnv
import numpy as np
import cv2


def file_name(file_dir="/model"):
    for root, dirs, files in os.walk(file_dir):
        for item in range(len(files)):
            files[item] = int(files[item][:-4])
        return max(files)


def load_model():
    # print "----loading previous models----"
    model_number = rl.get_file_number("model")
    while model_number > 0:
        try:
            model_number = rl.get_last_model("eval_ddpg")
            rl.eval_net = torch.load("models/eval_ddpg/" + str(model_number - 1) + ".pkl").cuda()
            rl.target_net = torch.load("models/target_ddpg/" + str(model_number - 1) + ".pkl").cuda()
            break
        except Exception as e:
            model_number -= 1
            continue


if __name__ == "__main__":

    MAX_EPISODES = 30000
    MAX_EP_STEPS = 50

    env = GazeboEnv()
    s_dim = 3
    a_dim = 3
    var = 3
    rl = DDPG()
    loss = 0

    for i in range(1, MAX_EPISODES):
        env.reset_world()
        print "*************ep={}***mc={}****************".format(i, rl.memory_counter)

        st = 0
        rw = 0

        observation = env.get_state()
        # 分成末端坐标和 rgbd

        while True:
            st += 1
            endg, view_state = observation
            action = rl.choose_action([endg, view_state])
            recent_end_goal, r, done = env.step(action)  # 执行一步
            observation_ = env.get_state()

            rl.store_transition(observation, action, r, observation_)

            if rl.memory_counter > 500:  # and st % 2 == 0:

                var *= .9995
                print(".....................learn.....................")
                loss = rl.learn()
            rw += r
            print "r = {},  done = {}".format(r, done)
            print "reward = {}".format(rw)
            if done or st >= MAX_EP_STEPS:
                log = open('log.txt', 'a')
                log.write("ep={}，reward={}，loss={}\n".format(i, rw,loss))
                break

            observation = observation_

        if i % 100 == 0:
            rl.save_model()
