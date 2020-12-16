import gym
from DQN.dqn00 import DQN
import os
import matplotlib.pyplot as plt

ENV_NAME = "CartPole-v0"
EPISODE = 800  # episode limitation
STEP = 200  # step limitation in an episode
# max_episode_steps=200 means that an episode automatically terminates after 200 steps.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    # initialize openAI GYm env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)
    step_ls = []
    total_step = 0
    for episode in range(1, EPISODE+1):
        # initialize task
        state = env.reset()
        for step in range(STEP):
            env.render()
            action = agent.egreedy_action(state)
            next_state, reward, done, _ = env.step(action)

            # The most important part is the reward function in the physical model.
            x, x_dot, theta, theta_dot = next_state
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            agent.perceive(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        total_step += step

        if episode % 20 == 0:
            avg_step = total_step/20
            step_ls.append(avg_step)
            print("Episode: %d, average step: %d, epsilon: %f" % (episode, avg_step, agent.epsilon))
            total_step = 0

    plt.plot(step_ls)
    plt.show()
    plt.savefig("result")


if __name__ == '__main__':
    main()
