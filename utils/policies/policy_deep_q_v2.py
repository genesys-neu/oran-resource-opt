import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils_for_policy import map_rb_to_action, map_action_to_rb, valid_actions, valid_actions_batch_tensor_version


class QNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Args:
            input_dim (int): state dimension.
            output_dim (int): number of actions.
            hidden_dim (int): hidden layer dimension (fully connected layer)
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        """
        Returns a Q value
        Args:
            state (torch.Tensor): state, 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q values, 2-D tensor of shape (n, output_dim)
        """
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        return x


class DeepQLearningAgent:
    def __init__(self, action_size=7, total_rb=17, max_num_users=10, penalty=0, seed=None, load=False, load_path_q=None):
        # problem setting
        self.action_size = action_size
        self.total_rb = total_rb
        self.max_num_users = max_num_users
        self.penalty = penalty
        # The Q network
        if load:
            self.dqn = QNetwork(5, action_size, 256)  # Q network
            state_dict = self.load_parameters(load_path_q)
            self.dqn.load_state_dict(state_dict)
        else:
            self.dqn = QNetwork(5, action_size, 256)  # Q network
        self.dqn_target = QNetwork(5, action_size, 256)  # Target Q network
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.loss_fn = torch.nn.MSELoss()  # loss function
        self.optim = torch.optim.Adam(self.dqn.parameters(), lr=0.01)  # optimizer for training
        self.gamma = 0.99
        self.trained_steps = 0
        self.target_update_period = 128
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)

    @staticmethod
    def load_parameters(load_path_q):
        assert load_path_q is not None, "Error! Path to load the Q table is undefined."
        state_dict = torch.load(load_path_q)
        return state_dict

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

        valid_actions_list = valid_actions(num_rb_mmtc, num_rb_urllc, num_rb_embb)

        self.dqn.eval()
        with torch.no_grad():
            state = torch.tensor([num_users_mmtc / self.max_num_users,
                                  num_users_urllc / self.max_num_users,
                                  num_users_embb / self.max_num_users,
                                  num_rb_mmtc / self.total_rb,
                                  num_rb_urllc / self.total_rb]).float().unsqueeze(0)
            scores = self.dqn(state)
        _, argmax = torch.max(scores.data[:, valid_actions_list], 1)
        action = valid_actions_list[int(argmax[0])]
        num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next = map_action_to_rb(num_rb_mmtc, num_rb_urllc, num_rb_embb, action)

        assert num_rb_mmtc_next + num_rb_urllc_next + num_rb_embb_next == self.total_rb, "The total number of RB of the input should be 17."
        assert num_rb_mmtc_next >= 1, "The number of RB for mmtc in the next step should be greater than or equal to 1"
        assert num_rb_urllc_next >= 1, "The number of RB for urllc in the next step should be greater than or equal to 1"
        assert num_rb_embb_next >= 1, "The number of RB for embb in the next step should be greater than or equal to 1"

        return num_rb_mmtc_next, num_rb_urllc_next, num_rb_embb_next

    def train(self, state_batch,
              action_batch,
              reward_batch,
              next_state_batch):
        """
        This function is used to train the Q network
        """
        self.dqn.train()
        current_q = self.dqn(state_batch).gather(1, action_batch.view(-1, 1).type(torch.int64))

        rb_mmtc = torch.round(next_state_batch[:, 3] * 17).to(torch.int)
        rb_urllc = torch.round(next_state_batch[:, 4] * 17).to(torch.int)
        rb_embb = 17 - rb_mmtc - rb_urllc
        valid_actions_mask = valid_actions_batch_tensor_version(rb_mmtc, rb_urllc, rb_embb)

        action_values_next = self.dqn_target(next_state_batch) * valid_actions_mask
        action_values_next[~valid_actions_mask] = -self.penalty
        next_q, _ = action_values_next.max(dim=1)
        next_q = next_q.view(-1, 1)

        # temp
        # state_test_1 = rows['num_mmtc_users'].iloc[0]
        # state_test_2 = rows['num_urllc_users'].iloc[0]
        # state_test_3 = rows['num_embb_users'].iloc[0]
        # state_test_4 = rows['pre_rb_mmtc'].iloc[0]
        # state_test_5 = rows['pre_rb_urllc'].iloc[0]
        # state_test_6 = 17 - rows['pre_rb_mmtc'].iloc[0] - rows['pre_rb_urllc'].iloc[0]

        assert -self.penalty <= torch.max(reward_batch) <= 1, "Error! reward should not be greater than one or less than the penalty."

        next_q = torch.clamp(next_q, min=-self.penalty / (1 - self.gamma), max=1 / (1 - self.gamma))

        Q_targets = reward_batch + self.gamma * next_q
        loss = self.loss_fn(current_q, Q_targets.detach())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.trained_steps += 1
        if self.trained_steps % self.target_update_period == 0:
            self.target_update()

    def target_update(self):
        # Update the target Q network (self.dqn_target) using the original Q network (self.dqn)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
