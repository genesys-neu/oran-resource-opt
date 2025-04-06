import numpy as np
import pandas as pd
import os

train_round = 3
algorithm = "TabularQ"  # "DeepQ" or "TabularQ" or "Bellman"
if train_round == 3:
    if algorithm == "DeepQ":
        path = "DeepQ_r2_from_DeepQ/"
    elif algorithm == "TabularQ":
        path = "TabularQ_r2_from_TabularQ/"
    elif algorithm == "Bellman":
        path = "DeepQ_r2_from_TabularQ/"

filename_from = "rl_dataset_train.csv"
filename_val = "rl_dataset_val.csv"
filename_val_add = "rl_dataset_val_add.csv"
filename_left = "rl_dataset_train_left.csv"


def main():
    # Read from dataset
    dataset_val = pd.read_csv(os.path.join(path, filename_val))
    print("Original Size of Validation Set: ", dataset_val.shape[0])

    rl_dataset = pd.read_csv(os.path.join(path, filename_from))
    print("Original Size of Training Set: ", rl_dataset.shape[0])

    # Read all user configurations
    common_rows = rl_dataset[['num_mmtc_users', 'num_urllc_users', 'num_embb_users']].drop_duplicates().sample(n=3, random_state=42)

    for row in rl_dataset.itertuples(index=True):
        is_present = common_rows[
            (common_rows['num_mmtc_users'] == row.num_mmtc_users) &
            (common_rows['num_urllc_users'] == row.num_urllc_users) &
            (common_rows['num_embb_users'] == row.num_embb_users)
            ].any(axis=None)  # Check if any row matches
        if is_present:
            dataset_val.loc[len(dataset_val)] = row[1:]
            rl_dataset = rl_dataset.drop(index=row.Index)

    # Reset the index after removing the row
    rl_dataset.reset_index(drop=True, inplace=True)

    if os.path.exists(os.path.join(path, filename_val_add)):
        os.remove(os.path.join(path, filename_val_add))
    if os.path.exists(os.path.join(path, filename_left)):
        os.remove(os.path.join(path, filename_left))
    print("Current Size of Training Set: ", rl_dataset.shape[0])
    rl_dataset.to_csv(os.path.join(path, filename_left), index=False)
    print("Current Size of Validation Set: ", dataset_val.shape[0])
    dataset_val.to_csv(os.path.join(path, filename_val_add), index=False)


if __name__ == "__main__":
    main()
    print("done")




