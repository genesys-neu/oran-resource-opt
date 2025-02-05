import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Parameters for analysis
confidence_level = 0.90  # Confidence level for convergence
threshold = 0.05         # Percentage of mean for the confidence interval (e.g., ±5%)

# Path to the dataset directory
base_dir = r"C:\Users\joshg\OneDrive - Northeastern University\IMPACT\RB_dataset"


# Function to compute confidence interval for the mean
def confidence_interval(data, confidence=0.90):
    n = len(data)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)  # Standard error of the mean
    z_score = 1.645  # For 90% confidence interval (z = 1.645)
    h = std_err * z_score
    return mean, mean - h, mean + h


# Function to calculate performance for a trial directory
def perf_func(row):
    ff = None
    try:
        # Extract required fields
        grant = row['sum_granted_prbs']
        req = row['sum_requested_prbs']
        dl_buffer = row['dl_buffer [bytes]']
        tx_brate = row['tx_brate downlink [Mbps]']
        prb = row['slice_prb']
        slice_id = row['slice_id']

        # Ensure all values are numeric and finite
        if not all(map(np.isfinite, [grant, req, dl_buffer, tx_brate, prb, slice_id])):
            raise ValueError("Non-numeric or infinite value detected.")

        # Calculate performance based on slice type
        if slice_id == 2:  # embb
            ff = (3 / 2 + tx_brate / 4 - dl_buffer * 8 / 1e6) / 3
            ff = max(0, min(1, ff))
        elif slice_id == 1:  # urllc
            if dl_buffer == 0:
                ff = 1.0
            elif dl_buffer > 0 and tx_brate == 0:
                ff = 0.0
            else:
                ff = max(0, 1 - dl_buffer * 8 / 1e6 / tx_brate)
        elif slice_id == 0:  # mmtc
            if prb == 0:
                ff = 0
            elif req == 0:
                ff = 1 / prb
            else:
                ff = min(1, grant / req)
        else:
            raise ValueError(f"Invalid slice_id: {slice_id}")

    except Exception as e:
        print(f"Skipping row due to error: {e}")
        return None

    return ff


def calculate_performance(trial_path):
    csv_files = [f for f in os.listdir(trial_path) if f.endswith(".csv")]
    if not csv_files:
        print(f"No .csv files found in {trial_path}")
        return None

    performance_list = []
    all_timestamps = None  # to store common timestamps

    # Step 1: Read all files and find common timestamps
    data_frames = []
    for csv_file in csv_files:
        file_path = os.path.join(trial_path, csv_file)
        data = pd.read_csv(file_path)
        required_columns = ['sum_granted_prbs', 'sum_requested_prbs', 'dl_buffer [bytes]',
                            'tx_brate downlink [Mbps]', 'slice_prb', 'slice_id', 'Timestamp']
        if not all(col in data.columns for col in required_columns):
            print(f"Missing required columns in {file_path}, skipping file.")
            continue

        # Store the data for later processing
        data_frames.append((file_path, data))

        # Collect all timestamps
        if all_timestamps is None:
            all_timestamps = set(data['Timestamp'])
        else:
            all_timestamps &= set(data['Timestamp'])  # intersection with existing timestamps

    if not all_timestamps:
        print(f"No common timestamps found across files in {trial_path}.")
        return None

    # Step 2: Process each file (filtering by common timestamps and calculating performance)
    for file_path, data in data_frames:
        # Ensure consistent slice_id by keeping the mode (most frequent value)
        mode_slice_id = data['slice_id'].mode()[0]
        data = data[data['slice_id'] == mode_slice_id]

        # Filter rows to keep only common timestamps across all files
        data = data[data['Timestamp'].isin(all_timestamps)]

        # Apply performance function and drop rows with None values
        data['performance'] = data.apply(perf_func, axis=1)
        valid_performance = data['performance'].dropna()

        if not valid_performance.empty:
            avg_performance = valid_performance.mean()
            performance_list.append(avg_performance)

    if not performance_list:
        print(f"No valid performance data in {trial_path}, skipping directory.")
        return None

    # Calculate final average performance
    trial_avg_performance = sum(performance_list) / len(performance_list)
    print(f"Average performance for {trial_path}: {trial_avg_performance:.4f}")
    return trial_avg_performance


# Traverse the directory structure
data = {}
performance_log = []
for method_dir in os.listdir(base_dir):
    method_path = os.path.join(base_dir, method_dir)
    if os.path.isdir(method_path):  # Ensure it's a directory
        print(f"Entering method directory: {method_dir}")
        for user_dir in os.listdir(method_path):
            if user_dir.startswith("users"):  # Only process directories that start with "users"
                user_path = os.path.join(method_path, user_dir)
                if os.path.isdir(user_path):  # Ensure it's a directory
                    print(f"Processing user configuration: {user_dir}")
                    trial_metrics = []
                    for trial_dir in os.listdir(user_path):
                        trial_path = os.path.join(user_path, trial_dir)
                        if trial_dir.startswith("Trial_") and os.path.isdir(trial_path):
                            try:
                                avg_perf = calculate_performance(trial_path)
                                if avg_perf is not None:  # Ensure only valid results are added
                                    trial_metrics.append(avg_perf)
                                    performance_log.append({
                                        "Method": method_dir,
                                        "User Combination": user_dir,
                                        "Trial": trial_dir,
                                        "Performance": avg_perf
                                    })
                            except Exception as e:
                                print(f"Error processing {trial_path}: {e}")

                    if trial_metrics:
                        user_combination = user_dir.replace("users", "")
                        if method_dir not in data:
                            data[method_dir] = {}
                        data[method_dir][user_combination] = trial_metrics

# Analyze convergence
results = []
for method, user_data in data.items():
    for user_combination, metrics in user_data.items():
        n_trials = len(metrics)
        if n_trials < 2:
            print(f"Insufficient data for {method} - {user_combination}, skipping additional trial computation.")
            results.append({
                "Method": method,
                "User Combination": user_combination,
                "Trials": n_trials,
                "Mean": metrics[0],
                "Median": metrics[0],
                "CI Width": np.nan,
                "Sufficient": False,
                "Additional Trials Needed": np.nan,
                "Std Dev": np.nan,
                "CV": np.nan,
                "Weighted Score": metrics[0] * 0.5
            })
            continue

        mean, lower_ci, upper_ci = confidence_interval(metrics, confidence=confidence_level)
        interval_width = upper_ci - lower_ci
        median = np.median(metrics)

        # Calculate standard deviation and CV
        std_dev = np.std(metrics, ddof=1)
        cv = std_dev / mean if mean != 0 else np.nan  # Avoid division by zero

        if mean == 0 or np.isnan(mean) or np.isnan(interval_width):
            print(f"Invalid metrics for {method} - {user_combination}, skipping additional trial computation.")
            results.append({
                "Method": method,
                "User Combination": user_combination,
                "Trials": n_trials,
                "Mean": mean,
                "Median": median,
                "CI Width": interval_width,
                "Sufficient": False,
                "Additional Trials Needed": np.nan,
                "Std Dev": std_dev,
                "CV": cv,
                "Weighted Score": mean * 0.5
            })
            continue

        is_sufficient = interval_width <= (threshold * mean)
        additional_trials_needed = None
        if not is_sufficient:
            try:
                additional_trials_needed = int(
                    ((1.645 * np.std(metrics, ddof=1)) / (threshold * mean))**2 - n_trials
                )
                additional_trials_needed = max(0, additional_trials_needed)
            except Exception as e:
                print(f"Error computing additional trials for {method} - {user_combination}: {e}")
                additional_trials_needed = np.nan

        results.append({
            "Method": method,
            "User Combination": user_combination,
            "Trials": n_trials,
            "Mean": mean,
            "Median": median,
            "CI Width": interval_width,
            "Sufficient": is_sufficient,
            "Additional Trials Needed": additional_trials_needed,
            "Std Dev": std_dev,
            "CV": cv,
            "Weighted Score": mean / (1 + cv)
        })

# Create a DataFrame for results
df_results = pd.DataFrame(results)
output_path = "RB_dataset_analysis.csv"
df_results.to_csv(output_path, index=False)
print(f"Analysis results saved to: {output_path}")

# Define the specific user combinations to include in the analysis
# valid_user_combinations = [
#     "_0_1_2", "_0_2_2", "_1_1_2", "_1_1_4", "_1_2_1",
#     "_1_2_3", "_1_2_5", "_1_3_4", "_1_3_5", "_2_3_4", "_3_2_3"
# ]

# Filter for user consistency analysis
user_consistency_data = []

# Aggregate metrics for each user combination across all methods
# user_metrics_aggregated = {user_comb: [] for user_comb in valid_user_combinations}
# Use defaultdict to automatically handle missing keys
user_metrics_aggregated = defaultdict(list)

# Populate the aggregated data
for method, user_data in data.items():
    for user_combination, metrics in user_data.items():
        user_metrics_aggregated[user_combination].extend(metrics)

# Calculate mean, median, and CV for each user combination
for user_combination, all_metrics in user_metrics_aggregated.items():
    if all_metrics:  # Ensure there are metrics to calculate
        n_trials = len(all_metrics)
        mean = np.mean(all_metrics)
        median = np.median(all_metrics)
        std_dev = np.std(all_metrics, ddof=1)
        cv = std_dev / mean if mean != 0 else np.nan  # Avoid division by zero
        user_consistency_data.append({
            "User Combination": user_combination,
            "CV": cv,
            "Mean": mean,
            "Median": median,
            "Trials": n_trials
        })

df_user_consistency = pd.DataFrame(user_consistency_data)
if not df_user_consistency.empty:
    df_user_consistency = df_user_consistency.sort_values(by="CV", ascending=True)
    df_user_consistency_path = "user_consistency.csv"
    df_user_consistency.to_csv(df_user_consistency_path, index=False)
    print(f"User consistency results saved to: {df_user_consistency_path}")
else:
    print("No valid user consistency data to save.")

# Filter for method consistency analysis
method_consistency_data = []
for method, user_data in data.items():
    all_metrics = []
    for user_combination, metrics in user_data.items():
        all_metrics.extend(metrics)
    if all_metrics:
        n_trials = len(all_metrics)
        mean = np.mean(all_metrics)
        median = np.median(all_metrics)
        std_dev = np.std(all_metrics, ddof=1)
        cv = std_dev / mean if mean != 0 else np.nan  # Avoid division by zero
        method_consistency_data.append({
            "Method": method,
            "CV": cv,
            "Mean": mean,
            "Median": median,
            "Trials": n_trials
        })

df_method_consistency = pd.DataFrame(method_consistency_data)
if not df_method_consistency.empty:
    df_method_consistency = df_method_consistency.sort_values(by="CV", ascending=True)
    df_method_consistency_path = "method_consistency.csv"
    df_method_consistency.to_csv(df_method_consistency_path, index=False)
    print(f"Method consistency results saved to: {df_method_consistency_path}")
else:
    print("No valid method consistency data to save.")
