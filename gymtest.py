import gym

env = gym.make('Birds-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(300):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode ended after {} timesteps with {} rewards".format(t+1, reward))
            break
    if not done:
        print("Episode not ended with {} timesteps".format(t+1))
env.close()
