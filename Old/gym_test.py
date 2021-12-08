import gym
env = gym.make('Acrobot-v1')
print('Acrobot-v1:')
print(env.observation_space)
print(env.action_space)
env = gym.make('Pendulum-v1')
print('Pendulum-v1:')
print(env.observation_space)
print(env.action_space)
env = gym.make('CartPole-v1')
print('CartPole-v1:')
print(env.observation_space)
print(env.action_space)
env = gym.make('MountainCar-v0')
print('MountainCar-v0')
print(env.observation_space)
print(env.action_space)
'''
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

env = gym.make('Pendulum-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
'''
