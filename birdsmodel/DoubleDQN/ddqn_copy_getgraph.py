import gym
from doubleDQN import DoubleDQN
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

agent = DoubleDQN(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01,
                  reward_decay=0.90,
                  e_greedy=0.90,
                  replace_target_iter=300,
                  memory_size=3000,
                  e_greedy_increment=0.001,
                  double_q=True)

total_steps = 0
reward_ls = []
steps = 800
x = np.linspace(0, 4 * np.pi, num=steps)
force_w = np.hstack([10 * np.cos(10 * x[:steps // 2]), -25 * np.sin(15 * x[steps // 2:])])
max_val = 0
print(force_w)
plt.plot(force_w)
plt.show()

for i_episode in range(50):
    observation = env.reset()
    ep_r = 0
    state_info = [[] for i in range(4)]
    for i in range(steps):
        # if i_episode % 50 == 0:
        #     env.render()
        # action = agent.choose_action(observation)
        env.external_force = force_w[i]
        # when there are no damping force, we need to set damping action = action[2] = 0
        action = 2
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward, observation_)
        for j in range(4):
            state_info[j].append(env.state[j])
        ep_r += reward
        # replay memory needs to store some experience at first to start training.
        # every learn will change epsilon
        if total_steps > 800:
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
        print("episode: %d, expected reward: %f" % (i_episode, sum(reward_ls[-20:]) / 20))

    reward_ls.append(ep_r)

now = datetime.now()
time = np.linspace(0, xtick * 0.02, num=xtick)
fig, axs = plt.subplots(2, 1)
axs[0].plot(time, best_state[0], time, best_state[2])
axs[0].set_ylabel('displacement')
axs[0].grid(True)

axs[1].plot(time, best_state[1], time, best_state[3])
axs[1].set_xlabel('steps')
axs[1].set_ylabel('velocity')
axs[1].grid(True)

plt.savefig("results_damping0.5_" + now.strftime('%Y-%m-%d-%H-%M') + '.jpg')
plt.show()
