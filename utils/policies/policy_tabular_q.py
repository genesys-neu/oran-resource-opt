import numpy as np
from .utils_for_policy import map_rb_to_action, map_action_to_rb


class TabularQLearningAgent:
    def __init__(self, action_size=7, total_rb=17, max_num_users=10, seed=None, load=False, load_path_q=None):
        # problem setting
        self.action_size = action_size
        self.total_rb = total_rb
        self.max_num_users = max_num_users
        # The Q table and visit counts
        if load:
            self.q_table = self.load_parameters(load_path_q)
        else:
            self.q_table = np.zeros((max_num_users + 1, max_num_users + 1, max_num_users + 1,
                                     total_rb, total_rb, action_size))

        self.visit_counts = np.zeros((max_num_users + 1, max_num_users + 1, max_num_users + 1,
                                      total_rb, total_rb, action_size), dtype=int)
        self.learning_rate = 0.1
        self.gamma = 0.99
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    @staticmethod
    def load_parameters(load_path_q):
        assert load_path_q is not None, "Error! Path to load the Q table is undefined."
        q_table = np.load(load_path_q)
        return q_table

    def policy(self, num_users_mmtc, num_users_urllc, num_users_embb, num_rb_mmtc, num_rb_urllc, num_rb_embb):
        """
        This function returns a RB configuration for the next step given the current RB configuration.
        Args:
            num_users_mmtc: the number of users in mmtc
            num_users_urllc: the number of users in urllc
            num_users_embb: the number of users in embb
            num_rb_mmtc: the number of RBs for mmtc in the current step
            num_rb_urllc: the number of RBs for urllc in the current step
            num_rb_embb: the number of RBs for embb in the current step
        Returns:
            num_rb_mmtc_next: the number of RBs for mmtc in the next step
            num_rb_urllc_next: the number of RBs for urllc in the next step
            num_rb_embb_next: the number of RBs for embb in the next step
        """
        assert num_rb_mmtc + num_rb_urllc + num_rb_embb == self.total_rb, "The total number of RB of the input should be 17."

        action = np.argmax(self.q_table[num_users_mmtc, num_users_urllc, num_users_embb,
                           num_rb_mmtc - 1, num_rb_urllc - 1, :])

        num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next = map_action_to_rb(num_rb_mmtc, num_rb_urllc, num_rb_embb,
                                                                                 action)

        assert num_rb_mmtc_next + num_rb_urllc_next + num_rb_embb_next == self.total_rb, "The total number of RB of the input should be 17."
        assert num_rb_mmtc_next >= 1, "The number of RB for mmtc in the next step should be greater than or equal to 1"
        assert num_rb_urllc_next >= 1, "The number of RB for urllc in the next step should be greater than or equal to 1"
        assert num_rb_embb_next >= 1, "The number of RB for embb in the next step should be greater than or equal to 1"

        return num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next

    def train(self, num_mmtc_users, num_urllc_users, num_embb_users,
              pre_rb_mmtc, pre_rb_urllc, pre_rb_embb,
              cur_action,
              reward,
              next_num_mmtc_users, next_num_urllc_users, next_num_embb_users,
              rb_mmtc, rb_urllc, rb_embb):
        """
        This function is used for the update of the Q table
        Args:
            - num_mmtc_users, num_urllc_users, num_embb_users, pre_rb_mmtc, pre_rb_urllc, pre_rb_embb: current state
            - cur_action: the current action
            - reward: the reward received
            - next_num_mmtc_users, next_num_urllc_users, next_num_embb_users, rb_mmtc, rb_urllc, rb_embb: next state
        """
        # if cur_action != 0:
        self.q_table[num_mmtc_users,
                     num_urllc_users,
                     num_embb_users,
                     pre_rb_mmtc - 1,
                     pre_rb_urllc - 1,
                     cur_action] = ((1 - self.learning_rate) * self.q_table[num_mmtc_users,
                                                                            num_urllc_users,
                                                                            num_embb_users,
                                                                            pre_rb_mmtc - 1,
                                                                            pre_rb_urllc - 1,
                                                                            cur_action]
                                    + self.learning_rate * (reward +
                                                            self.gamma * np.max(self.q_table[next_num_mmtc_users,
                                                                                             next_num_urllc_users,
                                                                                             next_num_embb_users,
                                                                                             rb_mmtc - 1,
                                                                                             rb_urllc - 1, :])))

    def update_visit_counts(self, num_mmtc_users, num_urllc_users, num_embb_users,
                            pre_rb_mmtc, pre_rb_urllc, cur_action):
        self.visit_counts[num_mmtc_users,
                          num_urllc_users,
                          num_embb_users,
                          pre_rb_mmtc - 1,
                          pre_rb_urllc - 1,
                          cur_action] += 1
