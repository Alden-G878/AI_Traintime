import random
import gym
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# This file is the human input version of l_test_model.py. It will use the user's keyboard inputs like the inputs that the
# AI would make. It will be used as the first part of my attempts to reduce training time, where I will train the AI on human
# generated inputs. I will then finish training without human input and compare it to an algorithm that has trained for the
# same total number of epochs. 

# There are a total of three actions, 0, 1, 2. I will have to test to see 

#from gym.cartpole.scores.score_logger import ScoreLogger

file_format = ".h5" # The file format that the model is saved as. If not h5, it would save as the SaveModel format, a tensorflow specific format
# However, this model is only compatible with tensorflow, to my knowedge (which isn't much, you should double check that)

checkpoint_path = "./models/human_input/checkpoint" 
# The checkpoint file (full filename would be 'checkpoint.h5') is saved every time the AI trains, 
# which is every time the enviornment progresses one frame

save_path = "./models/human_input/save_run" 
# The save_run file (full filename: 'save_run.h5') is updated (or created) every time a Run is completes.
# A run is completed when the 'terminal' variable is True, which is dictated by the OpenAI Gym library
# From what I've seen, a run is roughly 200 frames/steps, but I have not checked to see if that is 
# constant or it depends on something else.

print(tf.config.list_physical_devices())

ENV_NAME = "MountainCar-v0"
#ENV_NAME = "CartPole-v0"
# The ENV_NAME variable is the enviornment that the AI is training on. Check the OpenAI Gym website for the others.
# With this configuration, onmy [Box] (need to confirm name). Basically any enviornemnt that has a control scheme
# where there are no floating point values that scale the responce should work. I may assemble a list of the 
# enviornments, but the best place to check is the OpenAI Gym website.

# MountainCar and CartPole are the two models that I am planning to use, so I made two ENV_NAME assignments, and
# will comment each out as needed.



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
        #self.model.compile(optimizer="adam", loss="mse")#, lr=LEARNING_RATE)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=tf.keras.losses.MeanSquaredError())
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


def cartpole():
    env = gym.make(ENV_NAME)
    #score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        print(observation_space)
        while True:
            step += 1
            print("step: " + str(step))
            #env.render()
            # speed to ~0
            #print("env acc: " + str(env.state[1]))
            #if (env.state[0])
            if (env.state[1]==0):
                action = 1
            elif (env.state[1]<0):
                action = 0
            elif (env.state[1]>0):
                action = 2
            #action = int(input("0:L 1:N 2:R > "))#dqn_solver.act(state) #change this to be a human input, need to find action space and print it out
            #print(action)
            state_next, reward, terminal, info = env.step(action)
            #print(state_next)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            mem = dqn_solver.memory[step-1]
            #print(mem)
            save = tf.keras.Model(dqn_solver)
            #print(save)
            tf.keras.models.save_model(dqn_solver.model, checkpoint_path + file_format)
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                tf.keras.models.save_model(dqn_solver.model, save_path + str(run) + file_format)
                #score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay()


if __name__ == "__main__":
    cartpole()
'''
for/against this method
See code, github
'''