import os
import pandas as pd

train_round = 3
algorithm = "Bellman"  # "DeepQ" or "TabularQ" or "Bellman"
if train_round == 1:
    path_to_read = "RB_dataset/rl_dataset.csv"
    path_to_save = "RB_dataset/"
elif train_round == 2:
    # path_to_read = "RB_dataset_r2_TabularQ/rl_dataset.csv"
    # path_to_save = "RB_dataset_r2_TabularQ/"
    path_to_read = "RB_dataset_r2_DeepQ/rl_dataset.csv"
    path_to_save = "RB_dataset_r2_DeepQ/"
elif train_round == 3:
    if algorithm == "DeepQ":
        path_to_read = ["RB_dataset/rl_dataset.csv",
                        "RB_dataset_r2_DeepQ/rl_dataset.csv",
                        "DeepQ_r2_from_DeepQ/rl_dataset.csv"]
        path_to_save = "DeepQ_r2_from_DeepQ/"
    elif algorithm == "TabularQ":
        path_to_read = ["RB_dataset/rl_dataset.csv",
                        "RB_dataset_r2_TabularQ/rl_dataset.csv",
                        "TabularQ_r2_from_TabularQ/rl_dataset.csv"]
        path_to_save = "TabularQ_r2_from_TabularQ/"
    elif algorithm == "Bellman":
        path_to_read = ["RB_dataset/rl_dataset.csv",
                        "RB_dataset_r2_TabularQ/rl_dataset.csv",
                        "DeepQ_r2_from_TabularQ/rl_dataset.csv"]
        path_to_save = "DeepQ_r2_from_TabularQ/"
filename_train = "rl_dataset_train.csv"
filename_val = "rl_dataset_val.csv"


def main():
    train_val_ratio = 0.8
    if os.path.exists(os.path.join(path_to_save, filename_train)):
        os.remove(os.path.join(path_to_save, filename_train))

    if os.path.exists(os.path.join(path_to_save, filename_val)):
        os.remove(os.path.join(path_to_save, filename_val))

    # Read from dataset
    rl_dataset = []
    for i in range(len(path_to_read)):
        rl_dataset.append(pd.read_csv(path_to_read[i]))
    rl_dataset = pd.concat(rl_dataset, axis=0, ignore_index=True)

    # Divide the dataset into two parts, where one is for training and the other is for policy selection
    rl_dataset_train = rl_dataset.sample(frac=train_val_ratio, random_state=42)
    rl_dataset_val = rl_dataset.drop(rl_dataset_train.index)

    rl_dataset_train.to_csv(os.path.join(path_to_save, filename_train), index=False)
    rl_dataset_val.to_csv(os.path.join(path_to_save, filename_val), index=False)


if __name__ == "__main__":
    main()
    print("done")
