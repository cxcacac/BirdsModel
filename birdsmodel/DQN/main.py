import gym
from DQN.DQN_modified import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Birds-v0 is a discrete model, Birds-v1 is a continuous model.
env = gym.make('Birds-v0')
env = env.unwrapped

print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

agent = DeepQNetwork(n_actions=env.action_space.n,
                     n_features=env.observation_space.shape[0],
                     learning_rate=0.01,
                     reward_decay=0.90,
                     e_greedy=0.95,
                     replace_target_iter=500,
                     memory_size=10000,
                     e_greedy_increment=0.001)

total_steps = 0
reward_ls = []
steps = 500
x = np.linspace(0,  4 * np.pi, num=steps)
force_w = np.hstack([20 * np.cos(10 * x[:steps//2]), 30 * np.cos(15 * x[steps//2:])])

for i_episode in range(4000):
    observation = env.reset()
    ep_r = 0
    for i in range(steps):
        # if i_episode % 50 == 0:
        #     env.render()
        action = agent.choose_action(observation)
        env.external_force = force_w[i]
        # when there are no damping force, we need to set damping action = action[2] = 0
        # action = 2
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward, observation_)
        ep_r += reward
        # replay memory needs to store some experience at first to start training.
        # every learn will change epsilon
        if total_steps > 500:
            agent.learn()
        if done:
            break
        observation = observation_
        total_steps += 1

    if i_episode % 20 == 0:
        print("episode: %d, expected reward: %f, epsilon: %f" % (i_episode,
                                                                 sum(reward_ls[-20:]) / 20,
                                                                 agent.epsilon))

    reward_ls.append(ep_r)

now = datetime.now()
plt.ylim(0, 500)
plt.plot(reward_ls)
plt.savefig("results_memory10000_"+now.strftime('%Y-%m-%d-%H-%M')+'.jpg')
plt.show()

# agent.plot_cost()