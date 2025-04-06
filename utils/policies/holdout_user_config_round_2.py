import numpy as np
import pandas as pd
import os

path = ["RB_dataset_r2_TabularQ/", "RB_dataset_r2_DeepQ/"]
filename_from = "rl_dataset_train.csv"
filename_val = "rl_dataset_val.csv"
filename_val_add = "rl_dataset_val_add.csv"
filename_left = "rl_dataset_train_left.csv"
new_user_config_in_round2_from_Tabular = [[0, 1, 1],
                                          [0, 3, 2],
                                          [1, 0, 2],
                                          [1, 0, 5],
                                          [1, 1, 1],
                                          [1, 1, 5],
                                          [1, 1, 6],
                                          [1, 2, 2],
                                          [1, 2, 4],
                                          [1, 3, 1],
                                          [1, 3, 4],
                                          [1, 4, 2],
                                          [1, 4, 3],
                                          [2, 1, 4],
                                          [2, 2, 1],
                                          [2, 2, 3],
                                          [2, 3, 3],
                                          [2, 3, 5],
                                          [2, 4, 1],
                                          [2, 4, 2],
                                          [2, 4, 4],
                                          [3, 1, 4],
                                          [3, 2, 2],
                                          [3, 4, 1]]
new_user_config_in_round2_from_deep = [[1, 0, 2],
                                       [1, 1, 5],
                                       [1, 1, 6],
                                       [1, 2, 4],
                                       [1, 3, 3],
                                       [1, 4, 3],
                                       [2, 0, 2],
                                       [2, 1, 2],
                                       [2, 1, 4],
                                       [2, 2, 2],
                                       [2, 2, 3],
                                       [2, 3, 3],
                                       [2, 3, 5],
                                       [2, 4, 4],
                                       [3, 1, 4]]
# Convert the rows to tuples and use set intersection to find common rows
common_rows = (set(map(tuple, new_user_config_in_round2_from_Tabular))
               & set(map(tuple, new_user_config_in_round2_from_deep)))
# Convert back to lists
common_rows = [list(row) for row in common_rows]
common_rows = common_rows[2:5]


def main():

    for i in range(len(path)):
        # Read from dataset
        dataset_val_round_2 = pd.read_csv(os.path.join(path[i], filename_val))
        print("Original Size of Validation Set: ", dataset_val_round_2.shape[0])
        if os.path.exists(os.path.join(path[i], filename_val_add)):
            os.remove(os.path.join(path[i], filename_val_add))
        if os.path.exists(os.path.join(path[i], filename_left)):
            os.remove(os.path.join(path[i], filename_left))
        rl_dataset = pd.read_csv(os.path.join(path[i], filename_from))
        for row in rl_dataset.itertuples(index=False):
            if [row.num_mmtc_users, row.num_urllc_users, row.num_embb_users] in common_rows:
                dataset_val_round_2.loc[len(dataset_val_round_2)] = row
                condition = (rl_dataset == row).all(axis=1)
                rl_dataset = rl_dataset[~condition]
                # Reset the index after removing the row
                rl_dataset.reset_index(drop=True, inplace=True)
        rl_dataset.to_csv(os.path.join(path[i], filename_left), index=False)
        print("Current Size of Validation Set: ", dataset_val_round_2.shape[0])
        dataset_val_round_2.to_csv(os.path.join(path[i], filename_val_add), index=False)


if __name__ == "__main__":
    main()
    print("done")




