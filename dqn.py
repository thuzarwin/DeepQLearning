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
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1600, 512)
        self.fc2 = nn.Linear(512, actions)

    def forward(self, input_state):
        # input_state: [batch, 80x80x4]
        x = F.relu(self.conv1(input_state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
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
        self.observe = config['observe']
        self.steps_done = 0

        self.eval_net = ValueNet(self.actions)
        self.target_net = ValueNet(self.actions)
        self.optimizer = optim.RMSprop(self.eval_net.parameters(), self.learning_rate, weight_decay=0.99, momentum=0.9)
        self.loss_func = nn.MSELoss()

        self.replay_memory = deque()

        self.game_state = game.GameState()

    # Choose action according to the state
    def choose_action(self, state):
        prob = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1.*self.steps_done/self.epsilon_decay)
        self.steps_done += 1
        action_value = self.eval_net.forward(state).detach()
        if prob > eps_threshold:
            action_index = np.argmax(action_value.data.numpy())
        else:
            action_index = np.random.randint(0, self.actions)

        action = np.zeros(self.actions)
        action[action_index] = 1
        return action, action_value, eps_threshold

    # Apply action
    def apply_action(self, action):
        x, r, terminate = self.game_state.frame_step(action)
        x = cv2.cvtColor(cv2.resize(x, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, x = cv2.threshold(x, 1, 255, cv2.THRESH_BINARY)
        x = np.multiply(x, 1 / 255.0)
        x = np.reshape(x, (80, 80))
        return x, r, terminate

    # Store Memory
    def store_memory(self, s_t, a, r, s_t1, terminal):
        transition = (s_t, a, r, s_t1, terminal)
        if len(self.replay_memory) > self.replay_memory_size:
            self.replay_memory.popleft()
        self.replay_memory.append(transition)

    # Learn
    def learn(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
        minibatch = random.sample(self.replay_memory, self.batch_size)
        s_j_batch = Variable(torch.FloatTensor([d[0] for d in minibatch]))
        a_batch = Variable(torch.LongTensor(np.array([d[1] for d in minibatch]).astype(int)))
        r_batch = Variable(torch.FloatTensor([d[2] for d in minibatch]))
        s_j1_batch = Variable(torch.FloatTensor([d[3] for d in minibatch]))

        j1_batch = self.target_net.forward(s_j1_batch).detach()
        y_batch = []
        for i in range(self.batch_size):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(float(r_batch[i].data.numpy()[0]))
            else:
                y_batch.append(float((r_batch[i]+self.gamma*torch.max(j1_batch[i])).data.numpy()[0]))

        q_value = self.eval_net.forward(s_j_batch)
        q_value = q_value.gather(1, a_batch)
        q_value = q_value.view(self.batch_size, 1, 2)

        y_batch = Variable(torch.FloatTensor(y_batch))
        a_batch = a_batch.view(self.batch_size, 2, 1)

        a_batch = a_batch.type(torch.FloatTensor)
        q_ = torch.bmm(q_value, a_batch)
        #print(q_.shape)
        q_ = torch.squeeze(q_)
        y_batch = torch.squeeze(y_batch)
        #loss = (y_batch-q_value)
        #print(loss)
        loss = self.loss_func(q_, y_batch)

        print(loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    # play the Game
    def play(self):
        t = 0
        do_nothing = np.zeros(self.actions)
        do_nothing[0] = 1
        x_t, r_0, terminal = self.apply_action(do_nothing)
        s_t = [x_t, x_t, x_t, x_t]
        s_t = np.array(s_t)
        while "flapp bird" != "angry bird":
            action_t, q_value, eps = self.choose_action(Variable(torch.FloatTensor([s_t])))
            x_t1, r_t, terminal = self.apply_action(action_t)
            s_t1 = [s_t[1], s_t[2], s_t[3], x_t1]
            self.store_memory(s_t, action_t, r_t, s_t1, terminal)
            if t > self.observe:
                self.learn()

            s_t = s_t1
            t += 1

            action_index = np.argmax(action_t)

            print("TIMESTEP", t, "/ EPSILON", eps, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX:" , np.max(q_value.data.numpy()))


config = {
    'learning_rate': 0.01,
    'batch_size': 32,
    'replay_memory_size': 50000,
    'actions': 2,
    'gamma': 0.999,
    'epsilon_start': 0.95,
    'epsilon_end': 0.0001,
    'epsilon_decay': 200,
    'observe': 64

}


def play_angry_bird():

    dqn = DQN(config)

    dqn.play()


if __name__ == '__main__':

    play_angry_bird()










