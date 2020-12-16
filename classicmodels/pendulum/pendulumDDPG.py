import tensorflow as tf
import numpy as np
import gym

epsiode_steps = 500
lr_a = 0.001
lr_c = 0.002
gamma = 0.9
alpha = 0.01
memory = 10000
batch_size = 32
render = False


class DDPG(object):
    def __init__(self, no_of_actions, no_of_states, a_bound, ):
        # initialize the memory with shape as no of actions, no of states and our defined memory size
        self.memory = np.zeros((memory, no_of_states * 2 + no_of_actions + 1), dtype=np.float32)

        # initialize pointer to point to our experience buffer
        self.pointer = 0

        # initialize tensorflow session
        self.sess = tf.Session()

        # initialize the variance for OU process for exploring policies
        self.noise_variance = 3.0

        self.no_of_actions, self.no_of_states, self.a_bound = no_of_actions, no_of_states, a_bound,

        # placeholder for current state, next state and rewards
        self.state = tf.placeholder(tf.float32, [None, no_of_states], 's')
        self.next_state = tf.placeholder(tf.float32, [None, no_of_states], 's_')
        self.reward = tf.placeholder(tf.float32, [None, 1], 'r')

        # build the actor network which has separate eval(primary) and target network
        with tf.variable_scope('Actor'):
            self.a = self.build_actor_network(self.state, scope='eval', trainable=True)
            a_ = self.build_actor_network(self.next_state, scope='target', trainable=False)

        # build the critic network which has separate eval(primary) and target network
        with tf.variable_scope('Critic'):
            q = self.build_crtic_network(self.state, self.a, scope='eval', trainable=True)
            q_ = self.build_crtic_network(self.next_state, a_, scope='target', trainable=False)

        # initialize the network parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')

        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # update target value
        self.soft_replace = [
            [tf.assign(at, (1 - alpha) * at + alpha * ae), tf.assign(ct, (1 - alpha) * ct + alpha * ce)]
            for at, ae, ct, ce in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        # compute target Q value, we know that Q(s,a) = reward + gamma * Q'(s',a')
        q_target = self.reward + gamma * q_

        # compute TD error i.e  actual - predicted values
        td_error = tf.losses.mean_squared_error(labels=(self.reward + gamma * q_), predictions=q)

        # train the critic network with adam optimizer
        self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, name="adam-ink", var_list=self.ce_params)

        # compute the loss in actor network
        a_loss = - tf.reduce_mean(q)

        # train the actor network with adam optimizer for minimizing the loss
        self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        # initialize summary writer to visualize our network in tensorboard
        tf.summary.FileWriter("logs", self.sess.graph)

        # initialize all variables
        self.sess.run(tf.global_variables_initializer())

    # How do we select acion in DDPG? We select action by adding noise to the action space. We use
    # Ornstein-Uhlenbeck random process for generating noise

    def choose_action(self, s):
        a = self.sess.run(self.a, {self.state: s[np.newaxis, :]})[0]
        a = np.clip(np.random.normal(a, self.noise_variance), -2, 2)

        return a

    # then we define the function called learn where the actual training happens,
    # here we select a minibatch of states, actions, rewards and next state from the experience buffer
    # and we train actor and critic network

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(memory, size=batch_size)
        batch_transition = self.memory[indices, :]
        batch_states = batch_transition[:, :self.no_of_states]
        batch_actions = batch_transition[:, self.no_of_states: self.no_of_states + self.no_of_actions]
        batch_rewards = batch_transition[:, -self.no_of_states - 1: -self.no_of_states]
        batch_next_state = batch_transition[:, -self.no_of_states:]

        self.sess.run(self.atrain, {self.state: batch_states})
        self.sess.run(self.ctrain, {self.state: batch_states, self.a: batch_actions, self.reward: batch_rewards,
                                    self.next_state: batch_next_state})

    # we define a function store_transition which stores all the transition information in the buffer
    def store_transition(self, s, a, r, s_):
        trans = np.hstack((s, a, [r], s_))

        index = self.pointer % memory
        self.memory[index, :] = trans
        self.pointer += 1

        if self.pointer > memory:
            self.noise_variance *= 0.99995
            self.learn()

    # we define the function build_actor_network for builing our actor network
    def build_actor_network(self, s, scope, trainable):
        # Actor DPG
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 30, activation=tf.nn.tanh, name='l1', trainable=trainable)
            a = tf.layers.dense(l1, self.no_of_actions, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name="scaled_a")

            # followed by we define the function build_crtic_network which build our critic network

    def build_crtic_network(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.no_of_states, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.no_of_actions, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.tanh(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            q = tf.layers.dense(net, 1, trainable=trainable)
            return q


env = gym.make("Pendulum-v0")
env = env.unwrapped
env.seed(1)

no_of_states = env.observation_space.shape[0]
no_of_actions = env.action_space.shape[0]

a_bound = env.action_space.high

ddpg = DDPG(no_of_actions, no_of_states, a_bound)

total_reward = []
no_of_episodes = 300

for i in range(no_of_episodes):
    s = env.reset()
    ep_reward = 0

    for j in range(epsiode_steps):
        env.render()
        a = ddpg.choose_action(s)
        s_, r, done, info = env.step(a)
        ddpg.store_transition(s, a, r, s_)
        s = s_
        ep_reward += r

        if j == epsiode_steps - 1:
            # store the total rewards
            total_reward.append(ep_reward)
            # print rewards obtained per each episode
            print('Episode:', i, ' Reward: %i' % int(ep_reward))
            break