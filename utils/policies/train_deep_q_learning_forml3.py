import numpy as np
import copy
import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from policies.utils_for_policy import map_action_to_rb, map_rb_to_action
from train_tabular_q_learning_forml3 import cal_penalty
from policies.policy_deep_q import DeepQLearningAgent
from policies.policy_deep_q_large import DeepQLearningLargeAgent
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

pick = False
interpolation = True
large_network = True  # only for train_round >= 3
use_cuda = True
if use_cuda and torch.cuda.is_available() and large_network:
    device = "cuda"
else:
    device = "cpu"
train_round = 3
algorithm = "DeepQ"  # "DeepQ" or "Bellman"
if train_round == 2:
    data_from = "DeepQ"
    path_to_read = ["RB_dataset_r2_" + data_from + "/"]
    if pick:
        filename_train = ["rl_dataset_train_add_missing.csv"]
        filename_q = "dqn_forml3_r2_from_" + data_from + "_pick.pth"
    else:
        filename_train = ["rl_dataset_add_missing.csv"]
        filename_q = "dqn_forml3_r2_from_" + data_from + ".pth"
elif train_round == 3:
    if large_network:
        if pick:
            if interpolation:
                filename_q = "dqn_forml3_r3_" + "large_net_" + algorithm + "_pick.pth"
            else:
                filename_q = "dqn_forml3_r3_" + "large_net_" + algorithm + "_no_interpol" + "_pick.pth"
        else:
            if interpolation:
                filename_q = "dqn_forml3_r3_" + "large_net_" + algorithm + ".pth"
            else:
                filename_q = "dqn_forml3_r3_" + "large_net_" + algorithm + "_no_interpol" + ".pth"
    else:
        if pick:
            if interpolation:
                filename_q = "dqn_forml3_r3_" + algorithm + "_pick.pth"
            else:
                filename_q = "dqn_forml3_r3_" + algorithm + "_no_interpol" + "_pick.pth"
        else:
            if interpolation:
                filename_q = "dqn_forml3_r3_" + algorithm + ".pth"
            else:
                filename_q = "dqn_forml3_r3_" + algorithm + "_no_interpol" + ".pth"

    if algorithm == "DeepQ":
        path_to_read = ["DeepQ_r2_from_DeepQ/"]
        if pick:
            if interpolation:
                filename_train = ["rl_dataset_train_add_missing.csv"]
            else:
                filename_train = ["rl_dataset_train_left.csv"]
        else:
            if interpolation:
                filename_train = ["rl_dataset_add_missing.csv"]
            else:
                filename_train = ["rl_dataset_train.csv", "rl_dataset_val.csv"]
    elif algorithm == "Bellman":
        path_to_read = ["DeepQ_r2_from_TabularQ/"]
        if pick:
            if interpolation:
                filename_train = ["rl_dataset_train_add_missing.csv"]
            else:
                filename_train = ["rl_dataset_train_left.csv"]
        else:
            if interpolation:
                filename_train = ["rl_dataset_add_missing.csv"]
            else:
                filename_train = ["rl_dataset_train.csv", "rl_dataset_val.csv"]

path_to_save = "policies/"
sampling = True


def main():
    total_rb = 17
    action_size = 7
    max_num_users = 10
    penalty = 0
    batch_size = 128  # Batch size

    # Read from dataset
    rl_dataset = []
    if len(path_to_read) == 1 and len(filename_train) > 1:
        for i in range(len(filename_train)):
            rl_dataset.append(pd.read_csv(os.path.join(path_to_read[0], filename_train[i])))
    elif len(path_to_read) == len(filename_train):
        for i in range(len(path_to_read)):
            rl_dataset.append(pd.read_csv(os.path.join(path_to_read[i], filename_train[i])))
    else:
        assert False, "The length of path_to_read and filename_train does not match!"
    rl_dataset = pd.concat(rl_dataset, axis=0, ignore_index=True)

    # agent
    if large_network:
        agent = DeepQLearningLargeAgent(action_size=action_size, total_rb=total_rb, max_num_users=max_num_users,
                                        penalty=penalty, seed=42, device=device)
    else:
        agent = DeepQLearningAgent(action_size=action_size, total_rb=total_rb, max_num_users=max_num_users,
                                   penalty=penalty, seed=42)

    if sampling:
        invalid_action_counts = 0
        num_updates = 1000000
        for k in range(num_updates):
            if k % 1000 == 0:
                print("{} %".format(k / num_updates * 100))
                # test to see if it converges or not
                agent.dqn.eval()
                with torch.no_grad():
                    state = torch.tensor([3.0 / max_num_users,
                                          2.0 / max_num_users,
                                          3.0 / max_num_users,
                                          5 / total_rb,
                                          5 / total_rb]).float().unsqueeze(0).to(device)
                    scores = agent.dqn(state)
                print(scores.data)

            rows = rl_dataset.sample(n=batch_size, replace=True)

            actions = rows.apply(lambda row_: map_rb_to_action(int(row_.pre_rb_mmtc), int(row_.pre_rb_urllc),
                                                               int(row_.pre_rb_embb),
                                                               int(row_.rb_mmtc), int(row_.rb_urllc),
                                                               int(row_.rb_embb)), axis=1)

            rows['action'] = actions

            p = rows.apply(lambda row_: cal_penalty(int(row_.pre_rb_mmtc),
                                                    int(row_.pre_rb_urllc),
                                                    int(row_.pre_rb_embb),
                                                    int(row_.action), penalty) if not pd.isna(row_.action) else None,
                           axis=1)
            rows['penalty'] = p

            rewards = rows.apply(lambda row_: row_.avg_perf - row_.penalty if not pd.isna(row_.penalty) else None,
                                 axis=1)

            rows['reward'] = rewards

            rows = rows[['num_mmtc_users', 'num_urllc_users', 'num_embb_users',
                         'pre_rb_mmtc', 'pre_rb_urllc',
                         'action',
                         'rb_mmtc', 'rb_urllc',
                         'reward']]

            rows = rows.dropna(subset=['action'])
            if rows.shape[0] < batch_size:
                invalid_action_counts = invalid_action_counts + batch_size - rows.shape[0]

            # Normalization
            rows['num_mmtc_users'] = rows['num_mmtc_users'] / max_num_users
            rows['num_urllc_users'] = rows['num_urllc_users'] / max_num_users
            rows['num_embb_users'] = rows['num_embb_users'] / max_num_users
            rows['pre_rb_mmtc'] = rows['pre_rb_mmtc'] / total_rb
            rows['pre_rb_urllc'] = rows['pre_rb_urllc'] / total_rb
            rows['rb_mmtc'] = rows['rb_mmtc'] / total_rb
            rows['rb_urllc'] = rows['rb_urllc'] / total_rb

            state_batch = torch.tensor(rows[['num_mmtc_users', 'num_urllc_users', 'num_embb_users',
                                             'pre_rb_mmtc', 'pre_rb_urllc']].values).float().to(device)

            action_batch = torch.tensor(rows[['action']].values).int().to(device)
            next_state_batch = torch.tensor(rows[['num_mmtc_users', 'num_urllc_users', 'num_embb_users',
                                                  'rb_mmtc', 'rb_urllc']].values).float().to(device)
            reward_batch = torch.tensor(rows[['reward']].values).float().to(device)
            agent.train(state_batch, action_batch, reward_batch, next_state_batch)

        print("Number of samples dropped due to invalid action: ", invalid_action_counts)
    else:
        invalid_action_counts = 0
        num_iter = 2000
        for k in range(num_iter):
            print("iteration: ", k)
            invalid_action_counts = 0
            rl_dataset_shuffled = rl_dataset.sample(frac=1).reset_index(drop=True)
            # Iterate over the shuffled dataset
            state_batch_list = []
            action_batch_list = []
            next_state_batch_list = []
            reward_batch_list = []
            for row in rl_dataset_shuffled.itertuples():
                action = map_rb_to_action(int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb),
                                          int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))
                if action is None:
                    # print("Warning! An invalid sample occur. Sample dropped.")
                    invalid_action_counts += 1
                else:
                    p = cal_penalty(int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb), action, penalty)
                    state_batch_list.append([row.num_mmtc_users / max_num_users,
                                             row.num_urllc_users / max_num_users,
                                             row.num_embb_users / max_num_users,
                                             row.pre_rb_mmtc / total_rb,
                                             row.pre_rb_urllc / total_rb])
                    action_batch_list.append([action])
                    next_state_batch_list.append([row.num_mmtc_users / max_num_users,
                                                  row.num_urllc_users / max_num_users,
                                                  row.num_embb_users / max_num_users,
                                                  row.rb_mmtc / total_rb,
                                                  row.rb_urllc / total_rb])
                    reward = row.avg_perf - p
                    reward_batch_list.append([reward])
                    if len(state_batch_list) == batch_size:
                        state_batch = torch.tensor(state_batch_list).float().to(device)
                        action_batch = torch.tensor(action_batch_list).int().to(device)
                        next_state_batch = torch.tensor(next_state_batch_list).float().to(device)
                        reward_batch = torch.tensor(reward_batch_list).float().to(device)
                        agent.train(state_batch, action_batch, reward_batch, next_state_batch)
                        state_batch_list = []
                        action_batch_list = []
                        next_state_batch_list = []
                        reward_batch_list = []

            # test to see if it converges or not
            agent.dqn.eval()
            with torch.no_grad():
                state = torch.tensor([3.0 / max_num_users,
                                      2.0 / max_num_users,
                                      3.0 / max_num_users,
                                      5 / total_rb,
                                      5 / total_rb]).float().unsqueeze(0).to(device)
                scores = agent.dqn(state)
            print(scores.data)

        print("Number of samples dropped due to invalid action: ", invalid_action_counts)

    # Output variables
    # deep Q-network
    # Store network in cpu
    state_dict = agent.dqn.to("cpu").state_dict()
    torch.save(state_dict, os.path.join(path_to_save, filename_q))


if __name__ == "__main__":
    main()
    print("done")
