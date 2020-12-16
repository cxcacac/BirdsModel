import gym
from DDPG.DDPG_final import DDPG
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

MAX_EPISODES = 3000
MAX_EP_STEPS = 500
MEMORY_CAPACITY = 10000

RENDER = False
ENV_NAME = 'BirdsContinuous-v0'

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]

a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound,
            memory_size=10000,
            gamma=0.90,
            tau=0.01,
            batch_size=32,
            lr_a=0.001,
            lr_c=0.001
            )

x = np.linspace(0, 4 * np.pi, num=MAX_EP_STEPS)
force_w = np.hstack([20 * np.cos(10 * x[:MAX_EP_STEPS // 2]), 30 * np.cos(15 * x[MAX_EP_STEPS // 2:])])

variance = 4  # control exploration
reward_ls = []
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        # if i % 50 == 0:
        #     env.render()
        # add randomness to action selection for exploration
        # usage: s = np.random.normal(mu, sigma, 1000)
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, variance), -a_bound, a_bound)
        env.external_force = force_w[j]
        s_, r, done, info = env.step(a)
        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            variance *= .9995  # decay the action randomness
            ddpg.learn()
        s = s_
        ep_reward += r

        if done:
            # if ep_reward > -300:RENDER = True
            reward_ls.append(ep_reward)
            break

    if i % 20 == 0:
        print('Episode:', i, ' Reward: %.2f' % (sum(reward_ls[-20:]) / 20), 'Explore: %.2f' % variance)

now = datetime.now()
plt.ylim(0, 500)
plt.plot(reward_ls)
plt.savefig("results_" + now.strftime('%Y-%m-%d-%H-%M') + '.jpg')
plt.show()
