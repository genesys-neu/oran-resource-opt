import os
import pandas as pd

train_round = 2
if train_round == 1:
    path_to_read = "RB_dataset/rl_dataset.csv"
    path_to_save = "RB_dataset/"
elif train_round == 2:
    # path_to_read = "RB_dataset_r2_TabularQ/rl_dataset.csv"
    # path_to_save = "RB_dataset_r2_TabularQ/"
    path_to_read = "RB_dataset_r2_DeepQ/rl_dataset.csv"
    path_to_save = "RB_dataset_r2_DeepQ/"
filename_train = "rl_dataset_train.csv"
filename_val = "rl_dataset_val.csv"


def main():
    train_val_ratio = 0.8
    if os.path.exists(os.path.join(path_to_save, filename_train)):
        os.remove(os.path.join(path_to_save, filename_train))

    if os.path.exists(os.path.join(path_to_save, filename_val)):
        os.remove(os.path.join(path_to_save, filename_val))

    # Read from dataset
    rl_dataset = pd.read_csv(path_to_read)
    if train_round == 2:
        rl_dataset_1_round = pd.read_csv("RB_dataset/rl_dataset.csv")
        num_train = (rl_dataset.shape[0] + rl_dataset_1_round.shape[0]) * train_val_ratio - rl_dataset_1_round.shape[0]
        train_val_ratio = num_train / rl_dataset.shape[0]
    # Divide the dataset into two parts, where one is for training and the other is for policy selection
    rl_dataset_train = rl_dataset.sample(frac=train_val_ratio, random_state=42)
    rl_dataset_val = rl_dataset.drop(rl_dataset_train.index)

    rl_dataset_train.to_csv(os.path.join(path_to_save, filename_train), index=False)
    rl_dataset_val.to_csv(os.path.join(path_to_save, filename_val), index=False)


if __name__ == "__main__":
    main()
    print("done")
