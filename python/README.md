# README for Traffic Classification Modules

This subdirectory contains Python scripts for generating datasets and training/testing machine learning models for Open RAN (ORAN) traffic classification.

## 1. ORAN_models.py

### Overview
This script defines the machine learning models used for traffic classification in ORAN. Currently, it includes implementations for Transformer-based and CNN-based models.

### Features
- **Transformer Model**: Implements a multi-head self-attention mechanism for sequential data classification.
- **CNN Model**: Utilizes convolutional layers to extract spatial features from the input data.


## 2. create_dataset.py

### Overview
This script preprocesses datasets for training and testing the traffic classification models. The input data consists of the log files from multiple experiments.

### Features
- Searches a list of directories given in a text file for all .csv log files. 
- Concatenates and cleans the log files.
- Returns minimum and maximum values for each column in the log files.

### Usage
Before running the script, you may need to update the key variables:
```python
directories_file = 'directories.txt'  # Replace with your list_of_directories.txt file path
output_file = 'combined_metrics.csv'
stats_file = 'feature_stats.csv'
```

## 3. train_test.py

### Overview
This script manages the training and evaluation of the traffic classification models. It integrates dataset loading, model training, validation, and testing into a cohesive pipeline.

### Features
- Primarily supports Transformer model.
- Allows for dynamic selection of hyperparameters.
- Outputs detailed logs and metrics for performance evaluation.
- Normalizes the data using precomputed feature statistics.
- Splits the dataset into training, validation, and test sets.

### Usage
To train and test a model:
```bash
python train_test.py -f <training_dataset.csv> -t <identifier_for_this_model> -s <slice_length> --feature_stats <path/to/feature_stats.csv>
```

### Arguments
- `--file_input` or `-f`: Path to the file containing the consolidated data, `default='final_dataset.csv'`.
- `--trial_version` or `-t`: Description of the model used for naming.
- `--slice_length` or `-s`: Number of samples in time used for each classification decision, `default=32`.
- `--num_heads` or `-nh`: Number of transformer heads to use, `default=32`.
- `--pretrained_model` or `-p`: If you are fine-tuning a pre-trained model, provide the path to the pretrained model. Otherwise, training will be done from scratch.
- `--feature_status`: Path to the file containing the feature stats, `default='feature_stats.csv'`.


---

Feel free to reach out with any questions or issues related to the scripts!
