import gym
from policyGradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Birds-v0 is a discrete model, Birds-v1 is a continuous model.
env = gym.make('Birds-v0')
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

RL = PolicyGradient(n_actions=env.action_space.n,
                    n_features=env.observation_space.shape[0],
                    learning_rate=0.01,
                    reward_decay=0.90,
                    # output_graph=True,
                    )

total_steps = 0
reward_ls = []
steps = 500
flag = False
max_val = 0
x = np.linspace(0, 4 * np.pi, num=steps)
force_w = np.hstack([20 * np.cos(10 * x[:steps//2]), -30 * np.cos(15 * x[steps//2:])])
x_tick = 0

for i_episode in range(5000):
    observation = env.reset()
    step = 0
    ep_r = 0
    state_info = [[] for i in range(4)]
    while True and step < steps:
        # if i_episode % 50 == 0:
        #     env.render()
        # if RENDER:
        #     env.render()
        action = RL.choose_action(observation)
        env.external_force = force_w[step]
        observation_, reward, done, info = env.step(action)
        for i in range(4):
            state_info[i].append(env.state[i])
        RL.store_transition(observation, action, reward)
        ep_r += reward

        if done:
            # ep_rs_sum = sum(RL.ep_rs)
            # if 'running_reward' not in globals():
            #     running_reward = ep_rs_sum
            # else:
            #     running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            # if running_reward > DISPLAY_REWARD_THRESHOLD:
            #     flag = True
            # if i_episode % 20 == 0:
            #     print("episode:", i_episode, "  reward:", int(running_reward))
            vt = RL.learn()
            break
        observation = observation_
        step += 1

    # if ep_r > max_val:
    #     max_val = ep_r
    #     best_state = state_info
    #     if step < steps:
    #         xtick = step + 1
    #     else:
    #         xtick = step

    if i_episode % 20 == 0:
        print("episode: %d, expected reward: %f" % (i_episode, sum(reward_ls[-20:]) / 20))

    reward_ls.append(ep_r)

now = datetime.now()
# time = np.linspace(0, xtick*0.02, num=xtick)
# fig, axs = plt.subplots(2, 1)
# axs[0].plot(time, best_state[0], time, best_state[2])
# axs[0].set_ylabel('displacement')
# axs[0].grid(True)
#
# axs[1].plot(time, best_state[1], time, best_state[3])
# axs[1].set_xlabel('steps')
# axs[1].set_ylabel('velocity')
# axs[1].grid(True)
plt.plot(reward_ls)
plt.ylim(0,500)
plt.savefig("results" + now.strftime('%Y-%m-%d-%H-%M') + '.jpg')
plt.show()

# agent.plot_cost()
