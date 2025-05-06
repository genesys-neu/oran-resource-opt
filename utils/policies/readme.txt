1. Specify "path_to_read" and "path_to_save" in "process_data.py",
which are the path to read the original KPI data and the path to save the processed RL dataset.

2. Run "process_data.py", where we process the data such as removing inconsistent data
and rearrange the data in the form of RL data tuple.

3. Specify "train_round", "path_to_read", and "path_to_save" in "dataset_division_roundn.py"
"train_round" is the round of training
"path_to_read" is the paths of the datasets we currently have.
"path_to_save" is the path of folder to save the output datasets.

4. Run "dataset_division_roundn.py", 
which divides the dataset into the training set and validation set.

5. Specify "train_round" and "path" in "holdout_user_config_roundn.py".
"path" is the folder containing the training set and validation set.

6. Run "holdout_user_config_roundn.py",
which moves some user configurations from the training set to the validation set.

7. Run "add_virtual_samples_roundn.py" after specifying the "path",
which is the path of the training set.
This code will add virtual samples for unseen state using linear regression.

8. Run "train_deep_q_learning_forml3.py" and "train_tabular_q_learning_forml3.py".

9. Run "bellman_error_roundn.py" to calculate the Bellman error using the validation set.


The "Pick" variable in all the files: 
"Pick=True" means that we do not use the data in the validation set to train the policy.
We should set "Pick=True" in the previous steps.
"Pick=False" means that we use all the data (including trainig set and validation set) to train the policy.
After we pick a policy accoding to the Bellman error, 
we can set "Pick=False" to re-run "add_virtual_samples_roundn.py" and "train_deep_q_learning_forml3.py" (or "train_tabular_q_learning_forml3.py")

