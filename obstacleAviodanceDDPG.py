#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import os
import math
import random
import time

# ####################  hyper parameters  ####################

LR_A = 0.0005  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 1000
BATCH_SIZE = 120
N_STATES = 224 * 224 * 3


# ##############################  DDPG  ####################################
class ANet(nn.Module):  # ae(s)=a
    def __init__(self):
        super(ANet, self).__init__()
        self.resnet50 = models.resnet50(True)  # (1, 1000)
        self.fc1 = nn.Linear(1000 + 3, 520)
        self.fc2 = nn.Linear(520, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, rgb, pose):
        rgb = self.resnet50(rgb)
        x = torch.cat((rgb.float(), pose.float()), dim=1)
        a = self.fc1(x)
        x1 = F.relu(self.fc2(a))
        x2 = self.fc3(x1)
        return x2


class CNet(nn.Module):  # ae(s)=a
    def __init__(self):
        super(CNet, self).__init__()
        self.resnet50 = models.resnet50(True)  # (1, 1000)
        self.fc1 = nn.Linear(1000 + 6, 520)
        self.fc2 = nn.Linear(520, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, rgb, joint, ac):
        rgb = self.resnet50(rgb)
        x = torch.cat((rgb.float(), joint.float(), ac.float()), dim=1)
        a = self.fc1(x)
        x1 = F.relu(self.fc2(a))
        x2 = self.fc3(x1)
        return x2


class DDPG(object):
    def __init__(self):
        self.device_ids = [0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.memory = np.zeros((MEMORY_CAPACITY, (224 * 224 * 3 + 3) * 2 + 3 + 1))
        c = '0'
        for fi in os.listdir('memory/'):
            c = fi.split('.')[0]
            break
        self.memory_counter = int(c)  # 记忆库计数
        self.memory = np.load('memory/'+str(c)+'.npy')
        print "load memory {}.npy".format(c)
        self.Actor_eval = ANet()
        self.Actor_target = ANet()
        self.Critic_eval = CNet()
        self.Critic_target = CNet()
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        joint_view, image_view = s
        image_view = image_view.astype(np.float32)
        rgb_np = np.array(image_view).reshape(-1, 224, 224, 3)[:, :, :, :3]
        image_view_rgb = torch.from_numpy(rgb_np)
        image_view_rgb = image_view_rgb.permute(0, 3, 1, 2)
        joint_view = torch.from_numpy(np.array(joint_view).reshape(-1, 3))
        action = self.Actor_eval.forward(image_view_rgb, joint_view).detach()
        action = action.cpu().numpy()
        aa = np.clip(action, -0.1, 0.1)
        print "action {}---> clip {}".format(action, aa)
        return [aa[0][0], aa[0][1], aa[0][2]]

    def learn(self):
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_joint1 = torch.FloatTensor((b_memory[:, :3]).reshape(-1, 3))
        rgb_np = (b_memory[:, 3:N_STATES + 3]).reshape(-1, 224, 224, 3)[:, :, :, :3]
        b_rgb2 = torch.FloatTensor(rgb_np).permute(0, 3, 1, 2)
        # b_s = b_s1, b_s2
        # observation 2
        print '---step1--'
        b_a = torch.LongTensor((b_memory[:, N_STATES + 3:N_STATES + 6]).reshape(-1, 3).astype(float))
        b_r = torch.FloatTensor((b_memory[:, N_STATES + 6:N_STATES + 7]).reshape(-1, 1))
        b_joint_1 = torch.FloatTensor((b_memory[:, N_STATES + 7:N_STATES + 10]).reshape(-1, 3))
        rgb_np_ = (b_memory[:, -N_STATES:]).reshape(-1, 224, 224, 3)[:, :, :, :3]
        b_rgb_2 = torch.FloatTensor(rgb_np_).permute(0, 3, 1, 2)
        print '---step2--'
        a = self.Actor_eval(b_rgb2, b_joint1)
        q = self.Critic_eval(b_rgb2, b_joint1, a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        # print(q)
        # print(loss_a)
        print '---step3--'
        self.atrain.zero_grad()
        print '---step3.1--'
        loss_a.backward()
        print '---step3.2--'
        self.atrain.step()
        print '---step4--'
        a_ = self.Actor_target(b_rgb_2, b_joint_1)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(b_rgb_2, b_joint_1, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = b_r + GAMMA * q_  # q_target = 负的
        # print(q_target)
        q_v = self.Critic_eval(b_rgb2, b_joint1, b_a)
        # print(q_v)
        print '---step4.5--'
        td_error = self.loss_td(q_target, q_v)
        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()
        print '---step5--'
        self.soft_update(self.Actor_target, self.Actor_eval, TAU)
        self.soft_update(self.Critic_target, self.Critic_eval, TAU)
        print '---step6--'
        return td_error

    def store_transition(self, s, a, r, s_):
        a = np.array(a).reshape(-1, 3)
        if a[0][0] is np.nan:
            return
        s1, s2 = s
        s3, s4 = s_

        if str(type(s3)) == '<type \'numpy.float64\'>':
            s_ = s
        #  s3 == list == numpy.float todo
        s3, s4 = s_
        s1 = np.array(s1).reshape(-1, 3)
        s2 = np.array(s2).reshape(-1, 224 * 224 * 3)
        r = np.array(r).reshape(-1, 1)
        s3 = np.array(s3).reshape(-1, 3)
        s4 = np.array(s4).reshape(-1, 224 * 224 * 3)

        transition = np.hstack((s1, s2, a, r, s3, s4))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        if self.memory_counter % 100 == 0:
            print("saving memory")
            np.save("memory/" + str(self.memory_counter), self.memory)
            print "succeed saving memory, size {}".format(self.memory_counter)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )

    def save_model(self):
        c = 0
        for fi in os.listdir('model/Actor_eval_ddpg/'):
            c = fi.split('.')[0]
        model_number = c+1
        print("model_number:", model_number)
        torch.save(self.Actor_eval, "models/Actor_eval_ddpg/" + str(model_number) + ".pkl")
        torch.save(self.Actor_target, "models/Actor_target_ddpg/" + str(model_number) + ".pkl")
        torch.save(self.Critic_eval, "models/Critic_eval_ddpg/" + str(model_number) + ".pkl")
        torch.save(self.Critic_target, "models/Critic_target_ddpg/" + str(model_number) + ".pkl")


if __name__ == '__main__':
    # rl = DDPG()
    # rl.save_model()
    memory = np.zeros((1000, (224 * 224 * 3 + 3) * 2 + 3 + 1))
    np.save("memory/" + str(0), memory)
