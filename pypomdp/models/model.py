
from abc import abstractmethod
from util import draw_arg
import numpy as np


class Model(object):
    def __init__(self, env):
        """
        Expected attributes in env:
            model_name
            model_spec
            discount
            costs
            values
            states
            actions
            observations
            T
            Z
            R
        """
        for k, v in env.items():
            self.__dict__[k] = v

        self.curr_state = self.init_state or np.random.choice(self.states)

    @property
    def num_states(self):
        return len(self.states)

    @property
    def num_actions(self):
        return len(self.states)

    def gen_particles(self, n, prob=None):
        if prob is None:
            # by default use uniform distribution for particles generation
            prob = [1 / len(self.states)] * len(self.states)

        return [self.states[draw_arg(prob)] for i in range(n)]

    def get_legal_actions(self, state):
        """
        Simplest situation is every action is legal, but the actual model class
        may handle it differently according to the specific knowledge domain
        :param state:
        :return: actions selectable at the given state
        """
        #Tag problem, managing the possible actions

        # if 'Srv' in state:
        #     # Getting the column and row
        #     vertical = state[3:4]
        #     horiz = state[6:7]
        #
        #     # FIRST ROW
        #     # If the robot A is in the first row of the map, he shouldn't be able to move to the North
        #     f_actions = self.actions
        #     if vertical == 0:
        #         #del f_actions[0]
        #         f_actions.remove('North')
        #         # Cell 26, cant go north nor west
        #         if horiz == 5:
        #             #del f_actions[3]
        #             f_actions.remove('West')
        #         # Cell 28, cant go north nor east
        #         if horiz == 7:
        #             #del f_actions[2]
        #             f_actions.remove('East')
        #
        #     # SECOND ROW
        #     # If the robot A is in the second row may not be able to go west or east, depending on the cell
        #     if vertical == 1:
        #         # Cell 23, cant go west
        #         if horiz == 5:
        #             #del f_actions[3]
        #             f_actions.remove('West')
        #         # Cell 25, cant go east
        #         if horiz == 7:
        #             #del f_actions[2]
        #             f_actions.remove('East')
        #
        #     # THIRD ROW
        #     # If the robot A is in the third row may not be able to go west or east, depending on the cell
        #     if vertical == 2:
        #         # Cell 20, cant go west
        #         if horiz == 5:
        #             #del f_actions[3]
        #             f_actions.remove('West')
        #         # Cell 22, cant go east
        #         if horiz == 7:
        #             #del f_actions[2]
        #             f_actions.remove('East')
        #
        #     # FOURTH ROW
        #     # If the robot A is in the fourth row may not be able to go north, west or east, depending on the cell
        #     if vertical == 3:
        #         # Cells 10, 11, 12, 13, 14, 18 and 19 can't go north
        #         if horiz == 0 or horiz == 1 or horiz == 2 or horiz == 3 or horiz == 4 or horiz == 8 or horiz == 9:
        #             #del f_actions[0]
        #             f_actions.remove('North')
        #             # Cell 10, can't go west either
        #             if horiz == 0:
        #                 #del f_actions[3]
        #                 f_actions.remove('West')
        #             # Cell 19, can't go east either
        #             if horiz == 9:
        #                 #del f_actions[2]
        #                 f_actions.remove('East')
        #
        #     # FIFTH ROW / LAST ROW
        #     # If the robot A is in the last row of the map, he shouldn't be able to move to the South
        #     if vertical == 4:
        #         #del f_actions[1]
        #         f_actions.remove('South')
        #         # Cell 0, can't go west either
        #         if horiz == 0:
        #             #del f_actions[3]
        #             f_actions.remove('West')
        #
        #         # Cell 10, can't go east either
        #         if horiz == 9:
        #             #del f_actions[2]
        #             f_actions.remove('East')
        #
        #     return f_actions

        return self.actions

    def observation_function(self, action, state, obs):
        return self.Z.get((action, state, obs), 0.0)

    def transition_function(self, action, si, sj):
        return self.T.get((action, si, sj), 0.0)

    def reward_function(self, action='*', si='*', sj='*', obs='*'):
        return self.R.get((action, si, sj, obs), 0.0)

    def cost_function(self, action):
        if not self.costs:
            return 0
        return self.costs[self.actions.index(action)]

    def simulate_action(self, si, ai, debug=False):
        """
        Query the resultant new state, observation and rewards, if action ai is taken from state si

        si: current state
        ai: action taken at the current state
        return: next state, observation and reward
        """
        # get new state
        s_probs = [self.transition_function(ai, si, sj) for sj in self.states]
        state = self.states[draw_arg(s_probs)]

        # get new observation
        o_probs = [self.observation_function(ai, state, oj) for oj in self.observations]
        observation = self.observations[draw_arg(o_probs)]

        if debug:
            print('taking action {} at state {}'.format(ai ,si))
            print('transition probs: {}'.format(s_probs))
            print('obs probs: {}'.format(o_probs))

        # get new reward
        # reward = self.reward_function(ai, si, sj, observation) #  --- THIS IS MORE GENERAL!
        reward = self.reward_function(ai, si)   # --- THIS IS TMP SOLUTION!
        cost = self.cost_function(ai)

        return state, observation, reward, cost

    def take_action(self, action):
        """
        Accepts an action and changes the underlying environment state
        
        action: action to take
        return: next state, observation and reward
        """
        state, observation, reward, cost = self.simulate_action(self.curr_state, action)
        self.curr_state = state

        return state, observation, reward, cost

    def print_config(self):
        print("discount:", self.discount)
        print("values:", self.values)
        print("states:", self.states)
        print("actions:", self.actions)
        print("observations:", self.observations)
        print("")
        print("T:", self.T)
        print("")
        print("Z:", self.Z)
        print("")
        print("R:", self.R)
        print("")
