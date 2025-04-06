import numpy as np
import pandas as pd
import os
from policies.utils_for_policy import map_rb_to_action, valid_actions
from policies.policy_tabular_q import TabularQLearningAgent
from policies.policy_deep_q import DeepQLearningAgent
from policies.policy_deep_q_large import DeepQLearningLargeAgent
from policies.policy_stay import StayAgent
from train_tabular_q_learning_forml3 import cal_penalty
import torch

data_round = 3
alg_round = 4
data_from = "DeepQ"  # "DeepQ" or "TabularQ"  only for data_round=2
data_from_algorithm = "DeepQ"  # "DeepQ" or "TabularQ" or "Bellman"
agent_name = "DeepQ"  # "DeepQ" or "TabularQ"
interpolation = False
large_network = True  # only for data_round=3 and alg_round=4

if data_round == 1:
    path_to_read = ["RB_dataset/"]
    filename_val = ["rl_dataset_val_add.csv"]
elif data_round == 2:
    path_to_read = ["RB_dataset_r2_" + data_from + "/"]
    filename_val = ["rl_dataset_val_add.csv"]
elif data_round == 3:
    if data_from_algorithm == "DeepQ":
        path_to_read = ["DeepQ_r2_from_DeepQ/"]
    elif data_from_algorithm == "TabularQ":
        path_to_read = ["TabularQ_r2_from_TabularQ/"]
    elif data_from_algorithm == "Bellman":
        path_to_read = ["DeepQ_r2_from_TabularQ/"]
    filename_val = ["rl_dataset_val_add.csv"]

path_policy = "policies/"
if alg_round == 2:
    filename_q = "q_table_forml2.npy"
    filename_dqn = "dqn_forml2.pth"
elif alg_round == 3:
    filename_q = "q_table_forml2_r2_from_" + data_from + "_pick.npy"
    filename_dqn = "dqn_forml2_r2_from_" + data_from + "_pick.pth"
elif alg_round == 4:
    if interpolation:
        filename_q = "q_table_forml3_r3_" + data_from_algorithm + "_pick.npy"
        if large_network:
            filename_dqn = "dqn_forml3_r3_" + "large_net_" + data_from_algorithm + "_pick.pth"
        else:
            filename_dqn = "dqn_forml3_r3_" + data_from_algorithm + "_pick.pth"
    else:
        filename_q = "q_table_forml3_r3_" + data_from_algorithm + "_no_interpol_pick.npy"
        if large_network:
            filename_dqn = "dqn_forml3_r3_" + "large_net_" + data_from_algorithm + "_no_interpol_pick.pth"
        else:
            filename_dqn = "dqn_forml3_r3_" + data_from_algorithm + "_no_interpol_pick.pth"


def main():
    max_num_users = 10
    penalty = 0

    # Read from dataset
    rl_dataset = []
    for i in range(len(path_to_read)):
        rl_dataset.append(pd.read_csv(os.path.join(path_to_read[i], filename_val[i])))
    rl_dataset = pd.concat(rl_dataset, axis=0, ignore_index=True)

    # agent
    if agent_name == "TabularQ":
        agent = TabularQLearningAgent(seed=42,
                                      load=True, load_path_q=os.path.join(path_policy, filename_q))
    elif agent_name == "DeepQ":
        if large_network:
            agent = DeepQLearningLargeAgent(seed=42,
                                            load=True, load_path_q=os.path.join(path_policy, filename_dqn))
        else:
            agent = DeepQLearningAgent(seed=42,
                                       load=True, load_path_q=os.path.join(path_policy, filename_dqn))
    else:
        agent = StayAgent()

    avg_rel_bellman_error = 0.0
    avg_bellman_error = 0.0
    sample_count = 0

    for row in rl_dataset.itertuples():
        action = map_rb_to_action(int(row.pre_rb_mmtc), int(row.pre_rb_urllc), int(row.pre_rb_embb),
                                  int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))
        if action is not None:
            sample_count = sample_count + 1
            p = cal_penalty(row.pre_rb_mmtc, row.pre_rb_urllc, row.pre_rb_embb, action, penalty)
            reward = row.avg_perf - p
            if agent_name == "TabularQ":
                left = agent.q_table[int(row.num_mmtc_users), int(row.num_urllc_users), int(row.num_embb_users),
                                     int(row.pre_rb_mmtc) - 1, int(row.pre_rb_urllc) - 1, action]
                if alg_round == 4:
                    valid_actions_list = valid_actions(int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))
                    right = reward + agent.gamma * np.max(agent.q_table[int(row.num_mmtc_users),
                                                          int(row.num_urllc_users),
                                                          int(row.num_embb_users),
                                                          int(row.rb_mmtc) - 1, int(row.rb_urllc) - 1, valid_actions_list])
                else:
                    print("Warning: invalid action involves")
                    right = reward + agent.gamma * np.max(agent.q_table[int(row.num_mmtc_users),
                                                                        int(row.num_urllc_users),
                                                                        int(row.num_embb_users),
                                                                        int(row.rb_mmtc) - 1, int(row.rb_urllc) - 1, :])
                avg_rel_bellman_error = avg_rel_bellman_error + abs(left - right) / right
                avg_bellman_error = avg_bellman_error + abs(left - right)
            else:
                agent.dqn.eval()
                with torch.no_grad():
                    state = torch.tensor([row.num_mmtc_users / max_num_users,
                                          row.num_urllc_users / max_num_users,
                                          row.num_embb_users / max_num_users,
                                          row.pre_rb_mmtc / agent.total_rb,
                                          row.pre_rb_urllc / agent.total_rb]).float().unsqueeze(0)
                    scores = agent.dqn(state)
                    left = scores[0, action]
                    if alg_round == 4:
                        valid_actions_list = valid_actions(int(row.rb_mmtc), int(row.rb_urllc), int(row.rb_embb))

                        state = torch.tensor([row.num_mmtc_users / max_num_users,
                                              row.num_urllc_users / max_num_users,
                                              row.num_embb_users / max_num_users,
                                              row.rb_mmtc / agent.total_rb,
                                              row.rb_urllc / agent.total_rb]).float().unsqueeze(0)
                        scores = agent.dqn(state)
                        right = reward + agent.gamma * torch.max(scores.data[:, valid_actions_list], 1)[0]
                    else:
                        print("Warning: invalid action involves")
                        state = torch.tensor([row.num_mmtc_users / max_num_users,
                                              row.num_urllc_users / max_num_users,
                                              row.num_embb_users / max_num_users,
                                              row.rb_mmtc / agent.total_rb,
                                              row.rb_urllc / agent.total_rb]).float().unsqueeze(0)
                        scores = agent.dqn(state)
                        right = reward + agent.gamma * torch.max(scores.data, 1)[0]
                    avg_rel_bellman_error = avg_rel_bellman_error + abs(left - right) / right
                    avg_bellman_error = avg_bellman_error + abs(left - right)

    print("Relative Bellman Error: ", float(avg_rel_bellman_error) / sample_count)
    print("Bellman Error: ", float(avg_bellman_error) / sample_count)


if __name__ == "__main__":
    main()
    print("done")
