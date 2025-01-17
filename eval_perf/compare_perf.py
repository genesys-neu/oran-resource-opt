import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

test_round = 4
if test_round >= 1:
    path_to_read_r1 = "RB_dataset/rl_dataset.csv"
if test_round >= 2:
    path_to_read_r2_tabular = "RB_dataset_r2_TabularQ/rl_dataset_left.csv"
    path_to_read_r2_deep = "RB_dataset_r2_DeepQ/rl_dataset_left.csv"
if test_round >= 3:
    path_to_read_r3_tabular_from_tabular = "TabularQ_r2_from_TabularQ/rl_dataset.csv"
    path_to_read_r3_deep_from_tabular = "DeepQ_r2_from_TabularQ/rl_dataset.csv"
    path_to_read_r3_tabular_from_deep = "TabularQ_r2_from_DeepQ/rl_dataset.csv"
    path_to_read_r3_deep_from_deep = "DeepQ_r2_from_DeepQ/rl_dataset.csv"
if test_round >= 4:
    path_to_read_r4_tabular = "TabularQ_r3/rl_dataset.csv"
    path_to_read_r4_deep = "DeepQ_r3/rl_dataset.csv"
    path_to_read_r4_bellman_deep = "Bellman_r3_DeepQ_no_interpol/rl_dataset.csv"
    path_to_read_r4_bellman_tabular = "Bellman_r3_TabularQ_interpol/rl_dataset.csv"

path_to_read_expert = "Expert_config/rl_dataset.csv"

with_expert = True  # must be True currently
all_user_config = True
consider_only_proposed = False  # only when with_expert = True and all_user_config = True
if not with_expert:
    if all_user_config:
        path_to_save = "alg_test_up_to_round_{}_all_user_configs.csv".format(test_round)
    else:
        path_to_save = "alg_test_up_to_round_{}.csv".format(test_round)
else:
    if all_user_config:
        if consider_only_proposed:
            path_to_save = "alg_test_proposed_vs_expert_up_to_round_{}_all_user_configs.csv".format(test_round)
        else:
            path_to_save = "alg_test_with_expert_up_to_round_{}_all_user_configs.csv".format(test_round)
    else:
        path_to_save = "alg_test_with_expert_up_to_round_{}.csv".format(test_round)

max_num_users = 10


def main():
    # Read from dataset
    rl_dataset_total = []
    column_names = ['num_mmtc_users', 'num_urllc_users', 'num_embb_users']
    if test_round >= 1:
        rl_dataset_total.append(pd.read_csv(path_to_read_r1))
        column_names.append('round_1')
    if test_round >= 2:
        rl_dataset_total.append(pd.read_csv(path_to_read_r2_tabular))
        rl_dataset_total.append(pd.read_csv(path_to_read_r2_deep))
        column_names.append('round_2_tabular')
        column_names.append('round_2_deep')
    if test_round >= 3:
        rl_dataset_total.append(pd.read_csv(path_to_read_r3_tabular_from_tabular))
        rl_dataset_total.append(pd.read_csv(path_to_read_r3_deep_from_tabular))
        rl_dataset_total.append(pd.read_csv(path_to_read_r3_tabular_from_deep))
        rl_dataset_total.append(pd.read_csv(path_to_read_r3_deep_from_deep))
        column_names.append('round_3_tabular_from_tabular')
        column_names.append('round_3_deep_from_tabular')
        column_names.append('round_3_tabular_from_deep')
        column_names.append('round_3_deep_from_deep')
    if test_round >= 4:
        rl_dataset_total.append(pd.read_csv(path_to_read_r4_tabular))
        rl_dataset_total.append(pd.read_csv(path_to_read_r4_deep))
        rl_dataset_total.append(pd.read_csv(path_to_read_r4_bellman_deep))
        rl_dataset_total.append(pd.read_csv(path_to_read_r4_bellman_tabular))
        column_names.append('round_4_tabular')
        column_names.append('round_4_deep')
        column_names.append('round_4_bellman_deep')
        column_names.append('round_4_bellman_tabular')
    if with_expert:
        rl_dataset_total.append(pd.read_csv(path_to_read_expert))
        column_names.append('expert')

    # rl_dataset_0 = pd.read_csv(path_to_read_r1)
    # rl_dataset_1 = pd.read_csv(path_to_read_r2_tabular)
    # rl_dataset_2 = pd.read_csv(path_to_read_r2_deep)
    # rl_dataset_3 = pd.read_csv(path_to_read_r3_tabular_from_tabular)
    # rl_dataset_4 = pd.read_csv(path_to_read_r3_deep_from_tabular)
    # rl_dataset_5 = pd.read_csv(path_to_read_r3_tabular_from_deep)
    # rl_dataset_6 = pd.read_csv(path_to_read_r3_deep_from_deep)
    # rl_dataset_7 = pd.read_csv(path_to_read_expert)
    # rl_dataset_total = [rl_dataset_0, rl_dataset_1, rl_dataset_2, rl_dataset_3, rl_dataset_4, rl_dataset_5, rl_dataset_6,
    #                     rl_dataset_7]

    # results = pd.DataFrame(
    #     columns=['num_mmtc_users', 'num_urllc_users', 'num_embb_users',
    #              'round_1', 'round_2_tabular', 'round_2_deep',
    #              'round_3_tabular_from_tabular',
    #              'round_3_deep_from_tabular',
    #              'round_3_tabular_from_deep',
    #              'round_3_deep_from_deep',
    #              'expert'])
    results = pd.DataFrame(columns=column_names)

    for num_mmtc_u in range(max_num_users + 1):
        for num_urllc_u in range(max_num_users + 1):
            for num_embb_u in range(max_num_users + 1):
                # perf = [None, None, None, None, None, None, None, None]
                perf = []
                for i in range(len(rl_dataset_total)):
                    rl_dataset = rl_dataset_total[i]
                    small_dataset = rl_dataset[(rl_dataset.num_mmtc_users == num_mmtc_u) &
                                               (rl_dataset.num_urllc_users == num_urllc_u) &
                                               (rl_dataset.num_embb_users == num_embb_u)]
                    if small_dataset.shape[0] > 0:
                        if i != 0 and i != len(rl_dataset_total) - 1:
                            small_dataset = small_dataset[small_dataset['xapp']]
                        if small_dataset.shape[0] > 0:
                            rewards = small_dataset['pre_avg_perf']
                            perf.append(rewards.mean())
                        else:
                            perf.append(None)
                    else:
                        perf.append(None)
                assert len(perf) == len(rl_dataset_total), "Length of perf is not equal to length of rl_dataset_total"

                row = [num_mmtc_u, num_urllc_u, num_embb_u] + perf
                if all_user_config:
                    if consider_only_proposed:
                        if test_round >= 3:
                            if (perf[0] is not None or perf[1] is not None or perf[5] is not None
                                    or perf[len(rl_dataset_total) - 1] is not None):
                                results.loc[len(results)] = row
                        elif test_round >= 2:
                            if (perf[0] is not None or perf[1] is not None
                                    or perf[len(rl_dataset_total) - 1] is not None):
                                results.loc[len(results)] = row
                        else:
                            if perf[0] is not None or perf[len(rl_dataset_total) - 1] is not None:
                                results.loc[len(results)] = row

                    else:
                        if any(item is not None for item in perf):
                            results.loc[len(results)] = row
                else:
                    if None not in perf:
                        results.loc[len(results)] = row

    if os.path.exists(path_to_save):
        os.remove(path_to_save)
    results.to_csv(path_to_save, index=False)


if __name__ == "__main__":
    main()
    print("done")
