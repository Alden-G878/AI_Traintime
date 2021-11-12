import random
import gym
import numpy as np
import tensorflow as tf
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
import time



ENV_NAME = "MountainCar-v0"
env = gym.make(ENV_NAME)

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000000 #original: 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(optimizer="adam", loss="mse")#, lr=LEARNING_RATE)
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) #looks to be a basic mem write

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space) #random num gen
        q_values = self.model.predict(state) #uses neural net to predict best action
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0])) #Updates the q table
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


observation_space = env.observation_space.shape[0]
action_space = env.action_space.n
load_test_sm = DQNSolver(observation_space, action_space)
load_test_h5 = DQNSolver(observation_space, action_space)

print('init')
time1 = time.time()
load_test_sm.model = tf.keras.models.load_model('/Users/groverj/AdvInq/Current/models/checkpoint_test')
time2 = time.time()
print('SaveModel loaded')
print("ms taken, load model: " + str(time2-time1))
time1 = time.time()
mem_sm = load_test_sm.memory
time2 = time.time()
print("ms taken, access memory: " + str(time2-time1))
print('SaveModel mem')
time1 = time.time()
load_test_h5.model = tf.keras.models.load_model('/Users/groverj/AdvInq/Current/models/checkpoint_test2.h5')
time2 = time.time()
print('h5 loaded')
print("ms taken, load model: " + str(time2-time1))
time1 = time.time()
mem_h5 = load_test_h5.memory
time2 = time.time()
print("ms taken, access memory: " + str(time2-time1))
print('h5 mem')
