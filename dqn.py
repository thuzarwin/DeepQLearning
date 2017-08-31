# -*- coding: utf-8 -*-


import torch
import sys
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import cv2
import math
import random
sys.path.append("game/")
import wrapped_flappy_bird as game

class ValueNet(nn.Module):

    def __init__(self, actions):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(4, 32, 64, stride=2)
        self.conv3 = nn.Conv2d(3, 64, 64, stride=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, actions)

    def forward(self, input_state):
        # input_state: [batch, 80x80x4]

        x = F.relu(self.conv1(input_state))
        #x = self.pool(x)
        x = F.relu(self.conv2(x))
        #x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 1600)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class DQN(object):

    def __init__(self, config):
        self.learning_rate = config['learning_rate']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']
        self.replay_memory_size = config['replay_memory_size']
        self.actions = config['actions']
        self.steps_done = 0

        self.net = ValueNet(self.actions)
        self.optimizer = optim.RMSprop(self.net.parameters(), self.learning_rate, weight_decay=0.99, momentum=0.9)
        self.loss = nn.CrossEntropyLoss(size_average=False)

        self.replay_memory = deque()

        self.game_state = game.GameState()



    # Choose action according to the state
    def choose_action(self, state):
        prob = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1.*self.steps_done/self.epsilon_decay)
        if prob > eps_threshold:
            action_value = self.net.forward(state)
            action_index = np.argmax(action_value)
            action = np.zeros(self.actions)
            action[action_index] = 1
        else:
            action = np.random.randint(0, self.actions)
        return action

    # Apply action
    def apply_action(self, action):
        x, r, terminate = self.game_state.frame_step(action)
        x = cv2.cvtColor(cv2.resize(x, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
        return x, r, terminate

    # Store Memory
    def store_memory(self, s_t, a, r, s_t1):
        transition = np.hstack((s_t, a, r , s_t1))
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()
        self.replay_memory.append(transition)

    # Learn
    def learn(self):
        minibatch = random.sample(self.replay_memory, self.batch_size)
        s_j_batch = Variable(torch.FloatTensor([d[0] for d in minibatch]))
        a_batch = Variable(torch.LongTensor([d[1] for d in minibatch]))
        r_batch = Variable(torch.FloatTensor([d[2] for d in minibatch]))
        s_j1_batch = Variable(torch.FloatTensor([d[3] for d in minibatch]))

        j1_batch = self.net.forward(s_j_batch)
        y_batch = []
        for i in range(self.batch_size):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(r_batch[i])
            else:
                y_batch.append(r_batch[i]+self.gamma*np.max(j1_batch[i]))
        loss = self.loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    # play the Game
    def play(self):
        pass





config = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'replay_memory_size': 50000,
    'actions': 2,
    'gamma': 0.999,
    'epsilon_start': 0.90,
    'epsilon_end': 0.0001,
    'epsilon_decay': 200

}



# def train_dqn(learning_rate=0.001, gamma=0.99, batch_size=32, epsilon=0.0001, replay_memory=50000):
#
#     actions = 2
#     observe = 100000
#     explore = 2000000
#     dqn = DQN(actions=actions)
#     criterian = nn.CrossEntropyLoss(size_average=False)
#     optimizer = optim.RMSprop(dqn.parameters(), learning_rate,weight_decay=0.99, momentum=0.9)
#     #SGD(dqn.parameters(), lr=learning_rate)
#
#     game_state = game.GameState()
#
#     memory_D = deque()
#
#     do_nothing = np.zeros(actions)
#     do_nothing[0] = 1
#
#     def apply_action(action):
#         x, r, terminate = game_state.frame_step(action)
#         x = cv2.cvtColor(cv2.resize(x, (80, 80)), cv2.COLOR_BGR2GRAY)
#         ret, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
#
#         return x, r, terminate
#
#     def select_action(state):
#         output = dqn(state)[0]
#         action_t = np.zeros(actions)
#         if random.random() < epsilon:
#             print("----RANDOM ACRION----")
#             action_index = random.randrange(actions)
#             action_t[action_index] = 1
#         else:
#             action_index = np.argmax(output)
#             action_t[action_index] = 1
#         return action_t, action_index, output
#
#
#
#     x_t, r_0, isTerminated = apply_action(do_nothing)
#
#     s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
#     t = 0
#     while(1):
#         a_t, a_index, output = select_action(s_t)
#
#         x_t1, r_t, isTerminated = apply_action(a_t)
#         x_t1 = np.reshape(x_t1, (80, 80, 1))
#         s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
#
#         memory_D.append((s_t, a_t, r_t, s_t1))
#
#         if len(memory_D) > replay_memory:
#             memory_D.popleft()
#
#         if t > observe:
#             minibatch = random.sample(memory_D, batch_size)
#             s_j_batch = [d[0] for d in minibatch]
#             a_batch = [d[1] for d in minibatch]
#             r_batch = [d[2] for d in minibatch]
#             s_j1_batch = [d[3] for d in minibatch]
#
#             y_batch = []
#
#             j1_batch = dqn(s_j_batch)
#
#             for i in range(0, len(minibatch)):
#                 terminal = minibatch[i][4]
#                 # if terminal, only equals reward
#                 if terminal:
#                     y_batch.append(r_batch[i])
#                 else:
#                     y_batch.append(r_batch[i] + gamma * np.max(j1_batch[i]))
#
#             # train
#             loss =
#
#
#         s_t = s_t1
#         t += 1
#
#         if t <= observe:
#             state = "observe"
#         elif t > observe and t <= observe + explore:
#             state = "explore"
#         else:
#             state = "train"
#
#         print("TIMESTEP", t, "/ STATE", state, \
#             "/ EPSILON", epsilon, "/ ACTION", a_index, "/ REWARD", r_t, \
#             "/ Q_MAX %e" % np.max(output))







