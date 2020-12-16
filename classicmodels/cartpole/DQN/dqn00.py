import tensorflow as tf
import numpy as np
from collections import deque
import random

FINAL_EPSILON = 0.9
EPSILON_INCREMENT = 0.001
REPLAY_SIZE = 5000
BATCH_SIZE = 32
STEPS_START = 300
GAMMA = 0.9
HIDDEN_LAYER_NODES = 32
LEARNING_RATE = 0.001


class DQN:

    def __init__(self, env):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        # self.learning_step_counter = 0

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # None means any size, the shape of variable can be determined from the feed_dict.
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim], name="state")
        h_layer1 = self.add_layer(self.state_input, self.state_dim, HIDDEN_LAYER_NODES, activation_function=tf.nn.relu)
        h_layer2 = self.add_layer(h_layer1, HIDDEN_LAYER_NODES, HIDDEN_LAYER_NODES, activation_function=tf.nn.relu)
        self.Q_value = self.add_layer(h_layer2, HIDDEN_LAYER_NODES, self.action_dim)

    def add_layer(self, inputs, in_size, out_size, activation_function=None):
        """
        When establishing NN, or some structures we can determine the dimensions from matrix multiplication perspective.
        """
        weights = tf.Variable(tf.truncated_normal(shape=[in_size, out_size]))
        biases = tf.Variable(tf.constant(0.01, shape=[out_size]))
        exp = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = exp
        else:
            outputs = activation_function(exp)
        return outputs

    def create_training_method(self):
        # one-hot representation.
        # action_input must be two dimensional because of shape = [None, self.action_dim], number of axis = 2
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim], name="action")
        self.y_input = tf.placeholder(tf.float32, [None], name="input")

        q_value = tf.reduce_sum(
            tf.multiply(self.Q_value, self.action_input), reduction_indices=1)

        self.cost = tf.reduce_mean(tf.square(self.y_input - q_value))
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        """
        The action in this function should be in coordination with output np.argmax(Q) in self.egreedy()
        the function will learn the information(s, a, r, s'), and save it in replay buffer.
        only the buffer size > batch size, the training process begins.
        """
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append(
            (state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
        # self.learning_step_counter += 1

        if len(self.replay_buffer) > STEPS_START:
            self.train_Q_network()
            self.epsilon = self.epsilon + EPSILON_INCREMENT if self.epsilon < FINAL_EPSILON else FINAL_EPSILON

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]
        # if random seed < epsilon, we can use explore strategy to get more answers.
        if random.random() >= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

    def action(self, state):
        """
        This function is used to test the efficiency of Q NN.
        it select the maximum value of certain state, and return the corresponding action.
        """
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]
        return np.argmax(Q_value)

    def train_Q_network(self):
        """
        we use one NN to update itself.
        The process is same with Q learning, we use off-policy strategy to update the Q tables(NN)
        The expression would be like:
            Q(s,a) = Q(s,a) + gamma(r + GAMMA*max(Q(s', -)) - Q(s, a))
        """
        self.time_step += 1
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        # the feed_dict can be tuple
        state_batch, action_batch, reward_batch, next_state_batch, _ = zip(*minibatch)

        y_batch = []
        Q_value_batch = self.Q_value.eval(
            feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] +
                               GAMMA * np.max(Q_value_batch[i]))
        self.optimizer.run(
            feed_dict={
                self.y_input: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch
            }
        )
