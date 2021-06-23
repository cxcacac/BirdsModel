"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np
import random

logger = logging.getLogger(__name__)


class BirdsEnv(gym.Env):
    # class variables, two modes and 50 frames per seconds.
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # system constants
        self.gravity = 9.8
        self.head_mass = 0.5
        self.body_mass = 2.0

        self.spring_k = 20.0
        self.damping_c = 0.01
        self.tau = 0.02
        # The following parameter with unit m, m/s
        self.balance_head = 0.4
        self.balance_body = 0.0
        self.x_h_threshold = 0.8
        self.x_b_threshold = 0.8
        self.v_h_threshold = 0.5
        self.v_b_threshold = 2.0
        self.spring_length = 0.8
        # the parameter determine the size of window compared with realistic.
        self.height_threshold = 1.0
        # the force added on the body.
        self.external_force = 0.0
        # np.finfo(np.float32).max represent the maximum value of float32 type data.
        high = np.array([
            self.x_h_threshold*2 + self.balance_head,
            np.finfo(np.float32).max,
            self.x_b_threshold*2 + self.balance_body,
            np.finfo(np.float32).max])
        low = np.array([
            -self.x_h_threshold*2 + self.balance_head,
            -np.finfo(np.float32).max,
            -self.x_b_threshold*2 + self.balance_body,
            -np.finfo(np.float32).max])

        # {0,1,...,n-1}
        self.action_space = spaces.Discrete(5)
        self.damping_force = [-3, -1, 0, 1, 3]
        # the continuous control is for Birds-v1
        # self.action_space = spaces.Box(-self.damping_threshold, self.damping_threshold, shape=(1,))
        self.observation_space = spaces.Box(low, high)
        self._seed()
        self.reset()
        self.viewer = None

        self.steps_beyond_done = None

        # Just need to initialize the relevant attributes
        self._configure()

    def _configure(self, display=None):
        self.display = display

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action=None):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        # self.state: [head location, head velocity, body location, body velocity]
        l_h, v_h, l_b, v_b = self.state

        c = self.damping_c
        k = self.spring_k
        m_h = self.head_mass
        m_u = self.body_mass
        u_t = self.damping_force[action]
        spring_l = self.balance_head - self.balance_body
        force_w = self.external_force
        # force_w = random.uniform(-6, 6)
        # the dynamic equations need to consider the location of static equilibrium.
        h_acc = (u_t - c * (v_h - v_b) - k * (l_h - l_b - spring_l))/m_h
        b_acc = (force_w - u_t - c * (v_b - v_h) - k * (l_b - l_h + spring_l))/m_u

        v_h += h_acc * self.tau
        v_b += b_acc * self.tau
        # the changing velocity should have a limit, the changing velocity should not be so large.
        # v_h = np.clip(v_h, -self.v_h_threshold, self.v_h_threshold)
        # v_b = np.clip(v_b, -self.v_b_threshold, self.v_b_threshold)
        l_h += v_h * self.tau
        l_b += v_b * self.tau
        self.state = (l_h, v_h, l_b, v_b)

        self.delta_h = l_h - self.balance_head
        self.delta_b = l_b - self.balance_body

        done = bool(self.delta_h > self.x_h_threshold
                    or self.delta_b < -self.x_b_threshold
                    or l_h - l_b >= self.spring_length)

        # # the reward is related
        # r1 = self._eval(abs(self.delta_h), self.x_h_threshold, 0.01)
        r1 = 0
        r2 = self._eval(abs(v_h), self.v_h_threshold, 0.01)

        if not done:
            reward = r1 + r2
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = r1 + r2
        else:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive "
                    "'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _eval(self, diff, threshold, min_val):
        # the difference is the positive value.
        if diff >= threshold:
            return 0.0
        elif diff <= min_val:
            return 1.0
        else:
            gap = threshold - min_val
            return threshold / gap - 1.0 * diff / gap

    def _reset(self):
        """
        set the initial state randomly.
        using self.np_random.uniform(-self.range, self.range)
        """
        # the initial state should be legal, the height of head > the height of body.
        # self.state = self.np_random.uniform(low=-0.3, high=0.3, size=(4,))
        self.state = (self.balance_head,
                      0,
                      random.uniform(-0.3, 0.3)+self.balance_body,
                      0)
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600
        # establish the scale relations with real world, using physical units.
        world_width = self.height_threshold * 2
        scale = screen_width / world_width
        # rec1 represent the body, and rec2 represent the head.
        rec1width = 80.0
        rec1height = 30.0
        rec2width = 50.0
        rec2height = 30.0
        # to display the variation of displacement of rec1 and rec2, make them stay at the center of screen.
        rec1_bottom = 250

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height, display=self.display)

            l, r, t, b = -rec1width / 2, rec1width / 2, -rec1height / 2, rec1height / 2
            rec1 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            rec1.set_color(0, 0, 0)
            self.rec1trans = rendering.Transform()
            rec1.add_attr(self.rec1trans)
            self.viewer.add_geom(rec1)

            l, r, t, b = -rec2width / 2, rec2width / 2, -rec2height / 2, rec2height / 2
            rec2 = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            rec2.set_color(.8, .6, .4)
            self.rec2trans = rendering.Transform()
            rec2.add_attr(self.rec2trans)
            self.viewer.add_geom(rec2)

            # how to make line appear exactly once in one episode.
            line1 = rendering.Line((0, 0), (screen_width, 0))
            line1.set_color(0, 0, 0)
            self.line1trans = rendering.Transform()
            line1.add_attr(self.line1trans)
            self.viewer.add_geom(line1)

            line2 = rendering.Line((0, 0), (screen_width, 0))
            line2.set_color(.8, .6, .4)
            self.line2trans = rendering.Transform()
            line2.add_attr(self.line2trans)
            self.viewer.add_geom(line2)

        x = self.state
        rec1x = screen_width / 2.0
        rec1y = x[2] * scale + rec1_bottom
        rec2x = screen_width / 2.0
        rec2y = x[0] * scale + rec1_bottom

        # absolute transformations, reference the position of origin.
        self.rec1trans.set_translation(rec1x, rec1y)
        self.rec2trans.set_translation(rec2x, rec2y)
        self.line1trans.set_translation(0, self.balance_body * scale + rec1_bottom)
        self.line2trans.set_translation(0, self.balance_head * scale + rec1_bottom)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))
