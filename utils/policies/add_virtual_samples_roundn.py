import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from process_data import cal_avg_perf
from policies.utils_for_policy import map_rb_to_action, map_action_to_rb


train_round = 3
algorithm = "TabularQ"  # "DeepQ" or "TabularQ" or "Bellman"
if train_round == 3:
    if algorithm == "DeepQ":
        path = "DeepQ_r2_from_DeepQ/"
    elif algorithm == "TabularQ":
        path = "TabularQ_r2_from_TabularQ/"
    elif algorithm == "Bellman":
        path = "DeepQ_r2_from_TabularQ/"

pick = True
if pick:
    filename = ["rl_dataset_train_left.csv"]
    filename_add_missing = "rl_dataset_train_add_missing.csv"
    filename_add_missing_bootstrapping = "augmented_rl_dataset_train.csv"
    filename_visit = "original_visit_counts_train.npy"
else:
    filename = ["rl_dataset_train.csv", "rl_dataset_val.csv"]
    filename_add_missing = "rl_dataset_add_missing.csv"
    filename_add_missing_bootstrapping = "augmented_rl_dataset.csv"
    filename_visit = "original_visit_counts.npy"

total_rb = 17
max_num_users = 10
action_size = 7
num_new_samples = 10
std = 0.1
bootstrapping = False


def cal_avg_perf_from_slice_perf(num_mmtc_u, num_urllc_u, num_embb_u,
                                 perf_mmtc, perf_urllc, perf_embb):
    perf_list = [perf_mmtc, perf_urllc, perf_embb]
    num_user_list = [num_mmtc_u, num_urllc_u, num_embb_u]
    sum_perf = 0.0
    num_user = 0
    for i in range(3):
        if not pd.isna(perf_list[i]):
            sum_perf = sum_perf + perf_list[i] * num_user_list[i]
            num_user = num_user + num_user_list[i]
    avg_perf = sum_perf / num_user

    return avg_perf


def cal_perf(model_mmtc, model_urllc, model_embb, num_mmtc_u, num_urllc_u, num_embb_u, rb_mmtc_idx, rb_urllc_idx,
             visit_counts, small_dataset, is_pre):
    if is_pre:
        column1 = 'pre_rb_mmtc'
        column2 = 'pre_rb_urllc'
        column3 = 'pre_rb_embb'
        perf_column1 = 'pre_perf_mmtc'
        perf_column2 = 'pre_perf_urllc'
        perf_column3 = 'pre_perf_embb'
    else:
        column1 = 'rb_mmtc'
        column2 = 'rb_urllc'
        column3 = 'rb_embb'
        perf_column1 = 'perf_mmtc'
        perf_column2 = 'perf_urllc'
        perf_column3 = 'perf_embb'
    perf_mmtc = None
    perf_urllc = None
    perf_embb = None
    rb_mmtc = rb_mmtc_idx + 1
    rb_urllc = rb_urllc_idx + 1
    rb_embb = total_rb - rb_mmtc - rb_urllc
    assert rb_embb >= 1, "Error! The number of resource block in embb should be greater than or equal to 1."
    count2 = np.sum(visit_counts[rb_mmtc_idx, rb_urllc_idx, :])
    if count2 == 0:
        # interpolate mmtc
        if num_mmtc_u > 0:
            count3_1 = np.sum(visit_counts[rb_mmtc_idx, :, :])
            if count3_1 > 0:
                perf_mmtc = small_dataset[small_dataset[column1] == rb_mmtc][perf_column1].mean()
            else:
                perf_mmtc = model_mmtc.predict([[rb_mmtc]])[0]
                if perf_mmtc > 1:
                    perf_mmtc = 1.0
                if perf_mmtc < 0:
                    perf_mmtc = 0.0

        # interpolate urllc
        if num_urllc_u > 0:
            count3_2 = np.sum(visit_counts[:, rb_urllc_idx, :])
            if count3_2 > 0:
                perf_urllc = small_dataset[small_dataset[column2] == rb_urllc][perf_column2].mean()
            else:
                perf_urllc = model_urllc.predict([[rb_urllc]])[0]
                if perf_urllc > 1:
                    perf_urllc = 1.0
                if perf_urllc < 0:
                    perf_urllc = 0.0

        # interpolate embb
        if num_embb_u > 0:
            count3_3 = 0
            for rb_mmtc_idx_temp in range(total_rb):
                for rb_urllc_idx_temp in range(total_rb):
                    if rb_mmtc_idx_temp + rb_urllc_idx_temp + 2 == rb_mmtc + rb_urllc:
                        count3_3 = count3_3 + np.sum(
                            visit_counts[rb_mmtc_idx_temp, rb_urllc_idx_temp, :])
            if count3_3 > 0:
                perf_embb = small_dataset[small_dataset[column3] == rb_embb][perf_column3].mean()
            else:
                perf_embb = model_embb.predict([[rb_embb]])[0]
                if perf_embb > 1:
                    perf_embb = 1.0
                if perf_embb < 0:
                    perf_embb = 0.0
    else:
        tiny_set = small_dataset[(small_dataset[column1] == rb_mmtc)
                                 & (small_dataset[column2] == rb_urllc)]
        if num_mmtc_u > 0:
            perf_mmtc = tiny_set[perf_column1].mean()
        if num_urllc_u > 0:
            perf_urllc = tiny_set[perf_column2].mean()
        if num_embb_u > 0:
            perf_embb = tiny_set[perf_column3].mean()

    avg_perf = cal_avg_perf_from_slice_perf(num_mmtc_u, num_urllc_u, num_embb_u, perf_mmtc, perf_urllc, perf_embb)

    return perf_mmtc, perf_urllc, perf_embb, avg_perf


def main():
    dataset = []
    for i in range(len(filename)):
        dataset.append(pd.read_csv(os.path.join(path, filename[i])))
    dataset = pd.concat(dataset, axis=0, ignore_index=True)
    print("Original Dataset Size: ", dataset.shape[0])

    visit_counts = np.zeros((max_num_users + 1, max_num_users + 1, max_num_users + 1,
                             total_rb, total_rb, total_rb, total_rb), dtype=int)

    for row in dataset.itertuples():
        visit_counts[int(row.num_mmtc_users),
                     int(row.num_urllc_users),
                     int(row.num_embb_users),
                     int(row.pre_rb_mmtc) - 1,
                     int(row.pre_rb_urllc) - 1,
                     int(row.rb_mmtc) - 1,
                     int(row.rb_urllc) - 1] += 1

    np.save(os.path.join(path, filename_visit), visit_counts)

    for num_mmtc_u in range(max_num_users + 1):
        for num_urllc_u in range(max_num_users + 1):
            for num_embb_u in range(max_num_users + 1):
                count = np.sum(visit_counts[num_mmtc_u, num_urllc_u, num_embb_u, :, :, :])
                if count > 0:
                    print("Adding missing samples:", num_mmtc_u, num_urllc_u, num_embb_u)
                    small_dataset = dataset[(dataset.num_mmtc_users == num_mmtc_u) &
                                            (dataset.num_urllc_users == num_urllc_u) &
                                            (dataset.num_embb_users == num_embb_u)]
                    # linear regression
                    model_mmtc = None
                    model_urllc = None
                    model_embb = None
                    if num_mmtc_u > 0:
                        X = small_dataset[['rb_mmtc']]
                        y = small_dataset['perf_mmtc']
                        model_mmtc = LinearRegression()
                        model_mmtc.fit(X, y)
                    if num_urllc_u > 0:
                        X = small_dataset[['rb_urllc']]
                        y = small_dataset['perf_urllc']
                        X = X.drop(y[y.isnull()].index)
                        y = y.drop(y[y.isnull()].index)
                        model_urllc = LinearRegression()
                        assert (not X.isnull().values.any()), "Error! NaN encountered in rb_urllc when doing linear regression"
                        assert (not y.isnull().values.any()), "Error! NaN encountered in perf_urllc when doing linear regression"
                        model_urllc.fit(X, y)
                    if num_embb_u > 0:
                        X = small_dataset[['rb_embb']]
                        y = small_dataset['perf_embb']
                        model_embb = LinearRegression()
                        model_embb.fit(X, y)

                    for rb_mmtc_idx in range(total_rb):
                        for rb_urllc_idx in range(total_rb):
                            for pre_rb_mmtc_idx in range(total_rb):
                                for pre_rb_urllc_idx in range(total_rb):
                                    pre_rb_mmtc = pre_rb_mmtc_idx + 1
                                    pre_rb_urllc = pre_rb_urllc_idx + 1
                                    pre_rb_embb = total_rb - pre_rb_mmtc - pre_rb_urllc
                                    rb_mmtc = rb_mmtc_idx + 1
                                    rb_urllc = rb_urllc_idx + 1
                                    rb_embb = total_rb - rb_mmtc - rb_urllc
                                    if pre_rb_embb >= 1:
                                        if rb_embb >= 1:
                                            action = map_rb_to_action(pre_rb_mmtc, pre_rb_urllc, pre_rb_embb, rb_mmtc, rb_urllc, rb_embb)
                                            if action is not None:
                                                if visit_counts[num_mmtc_u, num_urllc_u, num_embb_u,
                                                                pre_rb_mmtc_idx, pre_rb_urllc_idx,
                                                                rb_mmtc_idx, rb_urllc_idx] == 0:
                                                    # print("Lack such data point: user config=({}, {}, {}), "
                                                    #       "pre_rb=({}, {}, {}), "
                                                    #       "rb=({}, {}, {})".format(num_mmtc_u, num_urllc_u, num_embb_u,
                                                    #                                    pre_rb_mmtc, pre_rb_urllc, pre_rb_embb,
                                                    #                                    rb_mmtc, rb_urllc, rb_embb))
                                                    visit_counts_temp = np.reshape(visit_counts[num_mmtc_u, num_urllc_u, num_embb_u,
                                                                                   :, :, :, :], (total_rb, total_rb,
                                                                                                 total_rb * total_rb))
                                                    pre_perf_mmtc, pre_perf_urllc, pre_perf_embb, pre_avg_perf \
                                                        = cal_perf(model_mmtc, model_urllc, model_embb,
                                                                   num_mmtc_u, num_urllc_u, num_embb_u,
                                                                   pre_rb_mmtc_idx, pre_rb_urllc_idx,
                                                                   visit_counts_temp, small_dataset, True)

                                                    visit_counts_temp = np.transpose(np.reshape(
                                                        visit_counts[num_mmtc_u, num_urllc_u, num_embb_u,
                                                                     :, :, :, :], (total_rb * total_rb,
                                                                                   total_rb, total_rb)), (1, 2, 0))
                                                    perf_mmtc, perf_urllc, perf_embb, avg_perf\
                                                        = cal_perf(model_mmtc, model_urllc, model_embb,
                                                                   num_mmtc_u, num_urllc_u, num_embb_u,
                                                                   rb_mmtc_idx, rb_urllc_idx,
                                                                   visit_counts_temp, small_dataset, False)

                                                    diff_perf = avg_perf - pre_avg_perf

                                                    assert not pd.isna(
                                                        diff_perf), "Error! diff_perf should not be nan or None."

                                                    dataset.loc[len(dataset)] = [None, num_mmtc_u, num_urllc_u, num_embb_u,
                                                                                 pre_rb_mmtc, pre_rb_urllc, pre_rb_embb,
                                                                                 rb_mmtc, rb_urllc, rb_embb,
                                                                                 pre_perf_mmtc, pre_perf_urllc, pre_perf_embb, pre_avg_perf,
                                                                                 perf_mmtc, perf_urllc, perf_embb, avg_perf,
                                                                                 diff_perf,
                                                                                 False]

    if os.path.exists(os.path.join(path, filename_add_missing)):
        os.remove(os.path.join(path, filename_add_missing))
    dataset.to_csv(os.path.join(path, filename_add_missing), index=False)

    print("Dataset size not including bootstrapping samples: {}".format(len(dataset)))

    # bootstrapping
    if bootstrapping:
        visit_counts = np.zeros((max_num_users + 1, max_num_users + 1, max_num_users + 1,
                                 total_rb, total_rb, total_rb, total_rb), dtype=int)

        for row in dataset.itertuples():
            visit_counts[int(row.num_mmtc_users),
                         int(row.num_urllc_users),
                         int(row.num_embb_users),
                         int(row.pre_rb_mmtc) - 1,
                         int(row.pre_rb_urllc) - 1,
                         int(row.rb_mmtc) - 1,
                         int(row.rb_urllc) - 1] += 1

        for num_mmtc_u in range(max_num_users + 1):
            for num_urllc_u in range(max_num_users + 1):
                for num_embb_u in range(max_num_users + 1):
                    count = np.sum(visit_counts[num_mmtc_u, num_urllc_u, num_embb_u, :, :, :])
                    if count > 0:
                        print("Bootstrapping: ", num_mmtc_u, num_urllc_u, num_embb_u)
                        small_dataset = dataset[(dataset.num_mmtc_users == num_mmtc_u) &
                                                (dataset.num_urllc_users == num_urllc_u) &
                                                (dataset.num_embb_users == num_embb_u)]
                        for rb_mmtc_idx in range(total_rb):
                            for rb_urllc_idx in range(total_rb):
                                for pre_rb_mmtc_idx in range(total_rb):
                                    for pre_rb_urllc_idx in range(total_rb):
                                        pre_rb_mmtc = pre_rb_mmtc_idx + 1
                                        pre_rb_urllc = pre_rb_urllc_idx + 1
                                        pre_rb_embb = total_rb - pre_rb_mmtc - pre_rb_urllc
                                        rb_mmtc = rb_mmtc_idx + 1
                                        rb_urllc = rb_urllc_idx + 1
                                        rb_embb = total_rb - rb_mmtc - rb_urllc
                                        if pre_rb_embb >= 1:
                                            if rb_embb >= 1:
                                                action = map_rb_to_action(pre_rb_mmtc, pre_rb_urllc, pre_rb_embb, rb_mmtc, rb_urllc, rb_embb)
                                                if action is not None:
                                                    assert visit_counts[num_mmtc_u, num_urllc_u, num_embb_u,
                                                                        pre_rb_mmtc_idx, pre_rb_urllc_idx,
                                                                        rb_mmtc_idx, rb_urllc_idx] > 0, "Error! Every transition should have at least one sample after interpolation."
                                                    tiny_set = small_dataset[(small_dataset['pre_rb_mmtc'] == pre_rb_mmtc)
                                                                             & (small_dataset[
                                                                                    'pre_rb_urllc'] == pre_rb_urllc)
                                                                             & (small_dataset['pre_rb_embb'] == pre_rb_embb)
                                                                             & (small_dataset['rb_mmtc'] == rb_mmtc)
                                                                             & (small_dataset['rb_urllc'] == rb_urllc)
                                                                             & (small_dataset['rb_embb'] == rb_embb)]
                                                    mean_pre_mmtc = tiny_set['pre_perf_mmtc'].mean()
                                                    mean_pre_urllc = tiny_set['pre_perf_urllc'].mean()
                                                    mean_pre_embb = tiny_set['pre_perf_embb'].mean()
                                                    mean_mmtc = tiny_set['perf_mmtc'].mean()
                                                    mean_urllc = tiny_set['perf_urllc'].mean()
                                                    mean_embb = tiny_set['perf_embb'].mean()
                                                    for i in range(num_new_samples):
                                                        pre_perf_mmtc = None
                                                        pre_perf_urllc = None
                                                        pre_perf_embb = None
                                                        perf_mmtc = None
                                                        perf_urllc = None
                                                        perf_embb = None
                                                        if num_mmtc_u > 0:
                                                            pre_perf_mmtc = np.clip(
                                                                np.random.normal(loc=mean_pre_mmtc, scale=std),
                                                                0, 1)
                                                            perf_mmtc = np.clip(np.random.normal(loc=mean_mmtc, scale=std),
                                                                                0, 1)
                                                        if num_urllc_u > 0:
                                                            pre_perf_urllc = np.clip(
                                                                np.random.normal(loc=mean_pre_urllc, scale=std),
                                                                0, 1)
                                                            perf_urllc = np.clip(
                                                                np.random.normal(loc=mean_urllc, scale=std),
                                                                0, 1)
                                                        if num_embb_u > 0:
                                                            pre_perf_embb = np.clip(
                                                                np.random.normal(loc=mean_pre_embb, scale=std),
                                                                0, 1)
                                                            perf_embb = np.clip(np.random.normal(loc=mean_embb, scale=std),
                                                                                0, 1)

                                                        pre_avg_perf = cal_avg_perf_from_slice_perf(num_mmtc_u, num_urllc_u, num_embb_u,
                                                                                                    pre_perf_mmtc, pre_perf_urllc, pre_perf_embb)

                                                        avg_perf = cal_avg_perf_from_slice_perf(num_mmtc_u, num_urllc_u, num_embb_u,
                                                                                                perf_mmtc, perf_urllc, perf_embb)

                                                        diff_perf = avg_perf - pre_avg_perf

                                                        assert not pd.isna(
                                                            diff_perf), "Error! diff_perf should not be nan or None."

                                                        dataset.loc[len(dataset)] = [None, num_mmtc_u, num_urllc_u,
                                                                                     num_embb_u,
                                                                                     pre_rb_mmtc, pre_rb_urllc, pre_rb_embb,
                                                                                     rb_mmtc, rb_urllc, rb_embb,
                                                                                     pre_perf_mmtc, pre_perf_urllc,
                                                                                     pre_perf_embb, pre_avg_perf,
                                                                                     perf_mmtc, perf_urllc, perf_embb, avg_perf,
                                                                                     diff_perf,
                                                                                     False]

        if os.path.exists(os.path.join(path, filename_add_missing_bootstrapping)):
            os.remove(os.path.join(path, filename_add_missing_bootstrapping))
        dataset.to_csv(os.path.join(path, filename_add_missing_bootstrapping), index=False)


if __name__ == "__main__":
    main()
    print('Done')
