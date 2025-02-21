import numpy as np
import copy
import math
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from policies.policy_tabular_q import TabularQLearningAgent
from policies.utils_for_policy import map_rb_to_action, map_action_to_rb

pick = False
interpolation = True
train_round = 3
algorithm = "Bellman"  # "TabularQ" or "Bellman"
if train_round == 2:
    data_from = "DeepQ"
    path_to_read = ["RB_dataset_r2_" + data_from + "/"]
    if pick:
        filename_train = ["rl_dataset_train_add_missing.csv"]
        filename_q = "q_table_forml3_r2_from_" + data_from + "_pick.npy"
        filename_visit = "visit_counts_forml3_r2_from_" + data_from + "_pick.npy"
    else:
        filename_train = ["rl_dataset_add_missing.csv"]
        filename_q = "q_table_forml3_r2_from_" + data_from + ".npy"
        filename_visit = "visit_counts_forml3_r2_from_" + data_from + ".npy"
elif train_round == 3:
    if pick:
        if interpolation:
            filename_q = "q_table_forml3_r3_" + algorithm + "_pick.npy"
            filename_visit = "visit_counts_forml3_r3_" + algorithm + "_pick.npy"
        else:
            filename_q = "q_table_forml3_r3_" + algorithm + "_no_interpol" + "_pick.npy"
            filename_visit = "visit_counts_forml3_r3_" + algorithm + "_no_interpol" + "_pick.npy"
    else:
        if interpolation:
            filename_q = "q_table_forml3_r3_" + algorithm + ".npy"
            filename_visit = "visit_counts_forml3_r3_" + algorithm + ".npy"
        else:
            filename_q = "q_table_forml3_r3_" + algorithm + "_no_interpol" + ".npy"
            filename_visit = "visit_counts_forml3_r3_" + algorithm + "_no_interpol" + ".npy"
    if algorithm == "TabularQ":
        path_to_read = ["TabularQ_r2_from_TabularQ/"]
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
sampling = False


def cal_penalty(pre_rb_mmtc, pre_rb_urllc, pre_rb_embb, action, penalty):
    """
    Calculate the reward penalty for training
    :param pre_rb_mmtc: number of RBs in mmtc
    :param pre_rb_urllc: number of RBs in urllc
    :param pre_rb_embb: number of RBs in embb
    :param action: the current action
    :param penalty: penalty of moving a RB
    :return: new_penalty: the calculated new penalty for training
    """
    assert action in range(7), "Error! 'action' should be 0, 1, 2, 3, 4, 5, or 6."
    """
    0: keep the current RB configuration
    1: mmtc -> urllc
    2: mmtc -> embb
    3: urllc -> mmtc
    4: urllc -> embb
    5: embb -> mmtc
    6: embb -> urllc
    """
    new_penalty = 0
    if action == 1:
        if pre_rb_mmtc >= 2:
            new_penalty = penalty
    elif action == 2:
        if pre_rb_mmtc >= 2:
            new_penalty = penalty
    elif action == 3:
        if pre_rb_urllc >= 2:
            new_penalty = penalty
    elif action == 4:
        if pre_rb_urllc >= 2:
            new_penalty = penalty
    elif action == 5:
        if pre_rb_embb >= 2:
            new_penalty = penalty
    elif action == 6:
        if pre_rb_embb >= 2:
            new_penalty = penalty

    return new_penalty


def main():
    total_rb = 17
    action_size = 7
    max_num_users = 10
    penalty = 0

    # Read from dataset
    rl_dataset = []
    for i in range(len(path_to_read)):
        rl_dataset.append(pd.read_csv(os.path.join(path_to_read[i], filename_train[i])))
    rl_dataset = pd.concat(rl_dataset, axis=0, ignore_index=True)

    # agent
    agent = TabularQLearningAgent(action_size=action_size, total_rb=total_rb, max_num_users=max_num_users, seed=42)

    if sampling:
        num_updates = 1000000
        invalid_action_counts = 0
        for k in range(num_updates):
            if k % 10000 == 0:
                print("{} %".format(k / num_updates * 100))
                # test to see if it converges or not
                print(np.max(agent.q_table))
            row = rl_dataset.sample(n=1, replace=True)
            action = map_rb_to_action(int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb),
                                      int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))
            if action is None:
                # print("Warning! An invalid sample occur. Sample dropped.")
                invalid_action_counts += 1
            else:
                p = cal_penalty(int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb), action, penalty)
                if k == 0:
                    agent.update_visit_counts(int(row.num_mmtc_users),
                                              int(row.num_urllc_users),
                                              int(row.num_embb_users),
                                              int(row.pre_rb_mmtc),
                                              int(row.pre_rb_urllc),
                                              action)

                reward = row.avg_perf - p

                agent.train(int(row.num_mmtc_users), int(row.num_urllc_users), int(row.num_embb_users),
                            int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb),
                            action,
                            reward,
                            int(row.num_mmtc_users), int(row.num_urllc_users), int(row.num_embb_users),
                            int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))

        print("Number of samples dropped due to invalid action: ", invalid_action_counts)
    else:
        invalid_action_counts = 0
        num_iter = 2000
        for k in range(num_iter):
            print("iteration: ", k)
            invalid_action_counts = 0
            rl_dataset_shuffled = rl_dataset.sample(frac=1).reset_index(drop=True)
            # Iterate over the shuffled dataset
            for row in rl_dataset_shuffled.itertuples():
                action = map_rb_to_action(int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb),
                                          int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))
                if action is None:
                    # print("Warning! An invalid sample occur. Sample dropped.")
                    invalid_action_counts += 1
                else:
                    p = cal_penalty(int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb), action, penalty)
                    if k == 0:
                        agent.update_visit_counts(int(row.num_mmtc_users),
                                                  int(row.num_urllc_users),
                                                  int(row.num_embb_users),
                                                  int(row.pre_rb_mmtc),
                                                  int(row.pre_rb_urllc),
                                                  action)

                    reward = row.avg_perf - p

                    agent.train(int(row.num_mmtc_users), int(row.num_urllc_users), int(row.num_embb_users),
                                int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb),
                                action,
                                reward,
                                int(row.num_mmtc_users), int(row.num_urllc_users), int(row.num_embb_users),
                                int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))

            # test to see if it converges or not
            print(np.max(agent.q_table))
            if k >= num_iter - 10:
                print(agent.q_table[3, 3, 3, :, :, 1])

        print("Number of samples dropped due to invalid action: ", invalid_action_counts)

    # Output variables
    # Q function
    q_values = agent.q_table
    # visit counts
    visit_counts = agent.visit_counts
    # Value function
    value_func = np.max(q_values, axis=-1)

    print(agent.q_table[3, 3, 3, :, :, 1])

    np.save(os.path.join(path_to_save, filename_q), q_values)
    np.save(os.path.join(path_to_save, filename_visit), visit_counts)


if __name__ == "__main__":
    main()
    print("done")
