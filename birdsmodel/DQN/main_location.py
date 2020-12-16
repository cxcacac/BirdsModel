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
y_body = np.hstack([0.3*np.cos(x[:steps//2]), 0.5*np.sin(2*x[steps//2:])])
max_val = 0

for i_episode in range(2000):
    observation = env.reset()
    ep_r = 0
    state_info = [[] for i in range(4)]
    for i in range(steps):
        # if i_episode % 50 == 0:
        #     env.render()
        action = agent.choose_action(observation)
        # env.state = np.array([env.state[0], env.state[1], y_body[i]+env.balance_body, env.state[3]])
        for j in range(4):
            state_info[j].append(env.state[j])
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

    if ep_r > max_val:
        max_val = ep_r
        best_state = state_info
        if i < steps:
            xtick = i + 1
        else:
            xtick = i

    if i_episode % 20 == 0:
        print("episode: %d, expected reward: %f, epsilon: %f" % (i_episode,
                                                                 sum(reward_ls[-20:]) / 20,
                                                                 agent.epsilon))

    reward_ls.append(ep_r)


now = datetime.now()
time = np.linspace(0, xtick*0.02, num=xtick)
fig, axs = plt.subplots(2, 1)
axs[0].plot(time, best_state[0], time, best_state[2])
axs[0].set_ylabel('displacement')
axs[0].grid(True)

axs[1].plot(time, best_state[1], time, best_state[3])
axs[1].set_xlabel('steps')
axs[1].set_ylabel('velocity')
axs[1].grid(True)

# now = datetime.now()
# plt.ylim(0, 500)
# plt.plot(reward_ls)
plt.savefig("results_body_y_excitation"+now.strftime('%Y-%m-%d-%H-%M')+'.jpg')
plt.show()
