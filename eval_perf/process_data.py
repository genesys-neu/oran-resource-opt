import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# from natsort import natsorted
import copy

path_to_read = r"C:\Users\joshg\OneDrive - Northeastern University\IMPACT\RB_dataset\Bellman_r3_DeepQ_no_interpol"
path_to_save = 'Bellman_r3_DeepQ_no_interpol/'
extension = 'metrics.csv'
filename = "rl_dataset.csv"
total_rb = 17


def cal_avg_perf(pre_perf_mmtc_list, pre_perf_urllc_list, pre_perf_embb_list,
                 perf_mmtc_list, perf_urllc_list, perf_embb_list):
    pre_perf_list = pre_perf_mmtc_list + pre_perf_urllc_list + pre_perf_embb_list
    perf_list = perf_mmtc_list + perf_urllc_list + perf_embb_list
    num_user_pre = 0
    pre_avg_perf = 0.0
    for perf in pre_perf_list:
        if perf is not None:
            num_user_pre += 1
            pre_avg_perf = pre_avg_perf + perf
    if num_user_pre == 0:
        pre_avg_perf = None
    else:
        pre_avg_perf = pre_avg_perf / num_user_pre

    num_user = 0
    avg_perf = 0.0
    for perf in perf_list:
        if perf is not None:
            num_user += 1
            avg_perf = avg_perf + perf
    if num_user == 0:
        avg_perf = None
    else:
        avg_perf = avg_perf / num_user

    return pre_avg_perf, avg_perf


def perf_func(row, traffic):
    ff = None
    grant = row['sum_granted_prbs']
    req = row['sum_requested_prbs']
    dl_buffer = row['dl_buffer [bytes]']
    tx_brate = row['tx_brate downlink [Mbps]']
    prb = row['slice_prb']
    if traffic == "embb":
        ff = (3/2 + tx_brate / 4 - dl_buffer * 8 / 1e6) / 3
        ff = max(0, min(1, ff))
    elif traffic == "urllc":
        if dl_buffer == 0:
            ff = 1.0
        elif dl_buffer > 0 and tx_brate == 0:
            ff = 0.0
        else:
            ff = max(0, 1 - dl_buffer * 8 / 1e6 / tx_brate)
    elif traffic == "mmtc":
        if req == 0:
            ff = 1 / prb
        else:
            ff = min(1, grant / req)
    else:
        print('Traffic type not detected')
        exit(1)

    return ff


def cal_avg_perf_intra_slice(perf_temp):
    num_user = 0
    avg_perf = 0.0
    for perf in perf_temp:  # Per user performance
        num_user += 1
        avg_perf = avg_perf + perf
    if num_user == 0:
        return None
    else:
        avg_perf = avg_perf / num_user
        return avg_perf


def cal_rb(df, idx, slice_users, slice_idx, user_config, trial):
    pre_rb_slice, rb_slice = None, None
    if len(slice_users[slice_idx]) != 0:
        pre_rb_temp = []
        rb_temp = []
        for i in slice_users[slice_idx]:
            if slice_idx == 2:
                pre_rb_temp.append(int((df[i].iloc[idx - 1].slice_prb + 1) // 3))
                rb_temp.append(int((df[i].iloc[idx].slice_prb + 1) // 3))
            else:
                pre_rb_temp.append(int(df[i].iloc[idx - 1].slice_prb // 3))
                rb_temp.append(int(df[i].iloc[idx].slice_prb // 3))

        if pre_rb_temp.count(pre_rb_temp[0]) != len(pre_rb_temp):  # if fluke happens, drop this sample
            print(
                "Warning! Fluke happens, user_config = {}, trial = {}, timestamp = {}, line = {}: "
                "multiple UEs in the same slice don't have the same PRB per slice. Sample dropped.".format(
                    user_config, trial,
                    int(df[0].iloc[idx - 1]['Timestamp']), idx + 1))
            normal_flag = False
            # pre_rb_slice = max(set(pre_rb_temp), key=pre_rb_temp.count)
            pre_rb_slice = -1  # drop this sample
        else:
            pre_rb_slice = pre_rb_temp[0]
        if rb_temp.count(rb_temp[0]) != len(rb_temp):  # if fluke happens, use mode
            print(
                "Warning! Fluke happens, user_config = {}, trial = {}, timestamp = {}, line = {}: "
                "multiple UEs in the same slice don't have the same PRB per slice. Sample dropped.".format(
                    user_config, trial,
                    int(df[0].iloc[idx]['Timestamp']), idx + 2))
            normal_flag = False
            # rb_slice = max(set(rb_temp), key=rb_temp.count)
            rb_slice = -1  # drop this sample
        else:
            rb_slice = rb_temp[0]
    return pre_rb_slice, rb_slice


def main():

    dataset = pd.DataFrame(columns=['Timestamp', 'num_mmtc_users', 'num_urllc_users', 'num_embb_users',
                                    'pre_rb_mmtc', 'pre_rb_urllc', 'pre_rb_embb',
                                    'rb_mmtc', 'rb_urllc', 'rb_embb',
                                    'pre_perf_mmtc', 'pre_perf_urllc', 'pre_perf_embb', 'pre_avg_perf',
                                    'perf_mmtc', 'perf_urllc', 'perf_embb', 'avg_perf',
                                    'diff_perf',
                                    'xapp'])
    user_configs = [name for name in os.listdir(path_to_read) if os.path.isdir(os.path.join(path_to_read, name))]
    for user_config in user_configs:
        trials = [name for name in os.listdir(os.path.join(path_to_read, user_config)) if
                  os.path.isdir(os.path.join(path_to_read, user_config, name))]

        for trial in trials:
            # if user_config != 'users_2_3_4' or trial != 'Trial_1':
            #     continue
            normal_flag = True
            # Read files
            csv_files = [f for f in os.listdir(os.path.join(path_to_read, user_config, trial)) if f.endswith(extension)]
            df = []
            for file in csv_files:
                df.append(pd.read_csv(os.path.join(path_to_read, user_config, trial, file)))

            # Remove the trial if the trial contains 0 slice_prb
            flag = False
            for each_df in df:
                if 0 in each_df.slice_prb.values:
                    flag = True
            if flag:
                print("Remove the trial because the trial contains 0 slice_prb: {} {}".format(user_config, trial))
                continue

            # Remove the files which have no data
            original_num_files = len(df)
            df = [each_df for each_df in df if len(each_df) > 0]
            if len(df) < original_num_files:
                print("Warning! Some file in the folder {}, {} has no data".format(user_config, trial))
                normal_flag = False
                continue

            # Calculate number of users in each slice and remove slice_id inconsistency
            num_mmtc_users = 0
            num_urllc_users = 0
            num_embb_users = 0
            slice_users = [[] for _ in range(3)]
            for i in range(len(df)):
                slice_id = df[i].slice_id.mode()
                if len(slice_id) > 1:
                    print("Warning! multiple modes for slice_id in the folder {}, {}".format(user_config, trial))
                    normal_flag = False
                slice_id = int(slice_id.iloc[0])
                slice_users[slice_id].append(i)
                if slice_id == 0:
                    num_mmtc_users = num_mmtc_users + 1
                elif slice_id == 1:
                    num_urllc_users = num_urllc_users + 1
                elif slice_id == 2:
                    num_embb_users = num_embb_users + 1
                else:
                    print('Traffic type not detected')
                    exit(1)
                # Remove slice_id inconsistency
                df[i] = df[i][df[i].slice_id == slice_id]

            # Check if the number of users in each slice is the same as the folder name
            if num_mmtc_users != int(user_config[6]) or num_urllc_users != int(user_config[8]) or num_embb_users != int(
                    user_config[10]):
                print("Warning! Inconsistent user config in the folder {}, {}".format(user_config, trial))
                normal_flag = False
                continue

            # Remove the Trial if there are more than one slices having 0 users
            if [num_mmtc_users, num_urllc_users, num_embb_users].count(0) >= 2:
                print("Remove the Trial because there are more than one slices having 0 users: {} {}".format(user_config, trial))
                normal_flag = False
                continue

            # Record whether xapp succeeded or not
            if ("xapp-logger.log" in os.listdir(os.path.join(path_to_read, user_config, trial))
                    and "warning" not in trial
                    and "Warning" not in trial):
                xapp = True
            else:
                xapp = False
                normal_flag = False
                print("No xapp involved in Trial: {}, {}".format(user_config, trial))

            # Remove timestamp inconsistency and num_ues inconsistency
            timestamp = df[0].Timestamp
            valid_timestamp = []
            for idx in range(timestamp.shape[0]):
                time = timestamp.iloc[idx]
                flag = True
                for i in range(len(df)):
                    if time not in df[i].Timestamp.values:
                        flag = False
                if flag:
                    valid_timestamp.append(time)
            for i in range(len(df)):
                df[i] = df[i][df[i].Timestamp.isin(valid_timestamp)]
                df[i] = df[i][df[i].num_ues == len(df)]

            # Calculate resource block and performance function
            for idx in range(df[0].shape[0]):
                if idx == 0:
                    continue
                timestamp = int(df[0].iloc[idx]['Timestamp'])

                # Calculate resource block
                # Slice 0
                pre_rb_mmtc, rb_mmtc = cal_rb(df, idx, slice_users, 0, user_config, trial)
                # Slice 1
                pre_rb_urllc, rb_urllc = cal_rb(df, idx, slice_users, 1, user_config, trial)
                # Slice 2
                pre_rb_embb, rb_embb = cal_rb(df, idx, slice_users, 2, user_config, trial)
                if (pre_rb_mmtc == -1 or rb_mmtc == -1
                        or pre_rb_urllc == -1 or rb_urllc == -1
                        or pre_rb_embb == -1 or rb_embb == -1):
                    continue

                if pre_rb_mmtc is None or rb_mmtc is None:
                    pre_rb_mmtc = total_rb - pre_rb_urllc - pre_rb_embb
                    rb_mmtc = total_rb - rb_urllc - rb_embb
                elif pre_rb_urllc is None or rb_urllc is None:
                    pre_rb_urllc = total_rb - pre_rb_mmtc - pre_rb_embb
                    rb_urllc = total_rb - rb_mmtc - rb_embb
                elif pre_rb_embb is None or rb_embb is None:
                    pre_rb_embb = total_rb - pre_rb_urllc - pre_rb_mmtc
                    rb_embb = total_rb - rb_urllc - rb_mmtc

                if pre_rb_mmtc + pre_rb_urllc + pre_rb_embb != total_rb:
                    print("warning! Total number of RBs should be 17: user_config={}, trial={}, timestamp={}, line={}. Sample Dropped".format(user_config, trial,
                                                                                                                                              int(df[0].iloc[idx - 1]['Timestamp']), idx + 1))
                    continue
                if rb_mmtc + rb_urllc + rb_embb != total_rb:
                    print("warning! Total number of RBs should be 17: user_config={}, trial={}, timestamp={}, line={}. Sample Dropped".format(user_config, trial,
                                                                                                                                              int(df[0].iloc[idx]['Timestamp']), idx + 2))
                    continue

                # assert pre_rb_mmtc + pre_rb_urllc + pre_rb_embb == total_rb, \
                #     "Error! Total number of RBs should be 17"
                # assert rb_mmtc + rb_urllc + rb_embb == total_rb, \
                #     "Error! Total number of RBs should be 17"

                # Calculate performance function
                # performance of slice 0: mmtc
                pre_perf_mmtc = None
                perf_mmtc = None
                pre_perf_mmtc_list = []
                perf_mmtc_list = []
                if num_mmtc_users != 0:
                    for i in slice_users[0]:
                        pre_perf_mmtc_list.append(perf_func(df[i].iloc[idx - 1], 'mmtc'))
                        perf_mmtc_list.append(perf_func(df[i].iloc[idx], 'mmtc'))
                    pre_perf_mmtc = cal_avg_perf_intra_slice(pre_perf_mmtc_list)
                    perf_mmtc = cal_avg_perf_intra_slice(perf_mmtc_list)

                # performance of slice 1: urllc
                pre_perf_urllc = None
                perf_urllc = None
                pre_perf_urllc_list = []
                perf_urllc_list = []
                if num_urllc_users != 0:
                    for i in slice_users[1]:
                        pre_perf_urllc_list.append(perf_func(df[i].iloc[idx - 1], 'urllc'))
                        perf_urllc_list.append(perf_func(df[i].iloc[idx], 'urllc'))
                    pre_perf_urllc = cal_avg_perf_intra_slice(pre_perf_urllc_list)
                    perf_urllc = cal_avg_perf_intra_slice(perf_urllc_list)

                # performance of slice 2: embb
                pre_perf_embb = None
                perf_embb = None
                pre_perf_embb_list = []
                perf_embb_list = []
                if num_embb_users != 0:
                    for i in slice_users[2]:
                        pre_perf_embb_list.append(perf_func(df[i].iloc[idx - 1], 'embb'))
                        perf_embb_list.append(perf_func(df[i].iloc[idx], 'embb'))
                    pre_perf_embb = cal_avg_perf_intra_slice(pre_perf_embb_list)
                    perf_embb = cal_avg_perf_intra_slice(perf_embb_list)

                pre_avg_perf, avg_perf = cal_avg_perf(pre_perf_mmtc_list, pre_perf_urllc_list, pre_perf_embb_list,
                                                      perf_mmtc_list, perf_urllc_list, perf_embb_list)
                diff_perf = avg_perf - pre_avg_perf

                dataset.loc[len(dataset)] = [timestamp, num_mmtc_users, num_urllc_users, num_embb_users,
                                             pre_rb_mmtc, pre_rb_urllc, pre_rb_embb,
                                             rb_mmtc, rb_urllc, rb_embb, 
                                             pre_perf_mmtc, pre_perf_urllc, pre_perf_embb, pre_avg_perf,
                                             perf_mmtc, perf_urllc, perf_embb, avg_perf,
                                             diff_perf,
                                             xapp]

            if normal_flag:
                print("Normal Trial: {}, {}".format(user_config, trial))

    if os.path.exists(os.path.join(path_to_save, filename)):
        os.remove(os.path.join(path_to_save, filename))
    dataset.to_csv(os.path.join(path_to_save, filename), index=False)


if __name__ == "__main__":
    main()
    print('Done')
