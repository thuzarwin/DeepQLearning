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
import random
sys.path.append("game/")
import wrapped_flappy_bird as game

class DQN(nn.Module):

    def __init__(self, actions):
        super(DQN, self).__init__()
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


def train_dqn(learning_rate=0.001, gamma=0.99, batch_size=32, epsilon=0.0001, replay_memory=5000):

    actions = 2
    dqn = DQN(actions=actions)
    criterian = nn.CrossEntropyLoss(size_average=False)
    optimizer = optim.RMSprop(dqn.parameters(), learning_rate,weight_decay=0.99, momentum=0.9)
    #SGD(dqn.parameters(), lr=learning_rate)

    game_state = game.GameState()

    memory_D = deque()

    do_nothing = np.zeros(actions)
    do_nothing[0] = 1

    def apply_action(action):
        x, r, terminate = game_state.frame_step(action)
        x = cv2.cvtColor(cv2.resize(x, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)

        return x, r, terminate

    def select_action(state):
        output = dqn(state)[0]
        action_t = np.zeros(actions)
        action_index = 0
        if random.random() < epsilon:
            print("----RANDOM ACRION----")
            action_index = random.randrange(actions)
            action_t[random.randrange(actions)] = 1
        else:
            action_index = np.argmax(output)
            action_t[action_index] = 1
        return action_t
    


    x_t, r_t, isTerminated = apply_action(do_nothing)

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)





    while(1):


















def test():
    print("Hello")





