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

#tf.debugging.set_log_device_placement(True)        # Debug - prints all device assignments made by tensorflow

#from gym.cartpole.scores.score_logger import ScoreLogger

file_format = ".h5" # The file format that the model is saved as. If not h5, it would save as the SaveModel format, a tensorflow format

checkpoint_path = "./models/checkpoint" 
# The checkpoint file (full filename would be 'checkpoint.h5') is saved every time the AI trains, 
# which is every time the enviornment progresses one frame

save_path = "./models/save_run" 
# The save_run file (full filename: 'save_run.h5') is updated (or created) every time a Run is completes.
# A run is completed when the 'terminal' variable is True, which is dictated by the OpenAI Gym library
# From what I've seen, a run is roughly 200 frames/steps, but I have not checked to see if that is 
# constant or it depends on something else.

print(tf.config.list_physical_devices())

ENV_NAME = "MountainCar-v0"
# The ENV_NAME variable is the enviornment that the AI is training on. Check the OpenAI Gym website for the others.
# With this configuration, onmy [Box] (need to confirm name). Basically any enviornemnt that has a control scheme
# where there are no floating point values that scale the responce should work. I may assemble a list of the 
# enviornments, but the best place to check is the OpenAI Gym website.

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
        while True:
            step += 1
            print("step: " + str(step))
            #print(tf.config.list_physical_devices())      #Debug - prints devices (COU, GPU(s), etc.) after every step
            #env.render()            # This should remove the rendering component of the openai gym, enabling this program ro be run on the command line
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
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
