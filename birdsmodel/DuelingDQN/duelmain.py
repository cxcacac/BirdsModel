import gym
from DuelingDQN.DuelDQN import DuelingDQN
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = env.action_space.n
OBSERVATION_N = env.observation_space.shape[0]

sess = tf.Session()
with tf.variable_scope('natural'):
    natural_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=OBSERVATION_N, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=False)

with tf.variable_scope('dueling'):
    dueling_DQN = DuelingDQN(
        n_actions=ACTION_SPACE, n_features=OBSERVATION_N, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, sess=sess, dueling=True)

sess.run(tf.global_variables_initializer())

steps = 500
x = np.linspace(0, 4 * np.pi, num=steps)
force_w = np.hstack([20 * np.cos(10 * x[:steps // 2]), 30 * np.cos(15 * x[steps // 2:])])

def train(agent):
    reward_ls = []
    total_steps = 0
    for i_episode in range(500):
        observation = env.reset()
        ep_r = 0
        for i in range(steps):
            action = agent.choose_action(observation)
            # env.external_force = force_w[i]
            observation_, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            agent.store_transition(observation, action, reward, observation_)
            ep_r += reward

            if total_steps > 500:
                agent.learn()
            if done:
                break
            observation = observation_
            total_steps += 1

        if i_episode % 50 == 0:
            print("episode: %d, expected reward: %f, epsilon: %f" % (i_episode,
                                                                     sum(reward_ls[-20:]) / 20,
                                                                     agent.epsilon))
        reward_ls.append(ep_r)
    return reward_ls


r_natural = train(natural_DQN)
r_dueling = train(dueling_DQN)

fig, axs = plt.subplots(2, 1)
axs[0].plot(r_natural)
axs[0].set_ylabel('natural')
axs[0].set_ylim(0, 500)
axs[0].grid(True)

axs[1].plot(r_dueling)
axs[1].set_xlabel('steps')
axs[1].set_ylabel('dueling')
axs[1].set_ylim(0, 500)
axs[1].grid(True)
now = datetime.now()
plt.savefig("results" + now.strftime('%Y-%m-%d-%H-%M') + '.jpg')
plt.show()
