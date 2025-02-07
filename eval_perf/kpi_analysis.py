import os
import pandas as pd
import numpy as np
from collections import defaultdict

# Path to the dataset directory
base_dir = r"C:\Users\joshg\OneDrive - Northeastern University\IMPACT\RB_dataset"
output_file = "kpi_overhead.csv"  # Output file

# Define fixed and selectable KPIs
fixed_kpis = ["Timestamp", "num_ues", "IMSI", "RNTI", "slicing_enabled",
              "slice_id", "slice_prb", "power_multiplier", "scheduling_policy"]

selectable_kpis = ["dl_mcs", "dl_n_samples", "dl_buffer [bytes]", "tx_brate downlink [Mbps]",
                   "tx_pkts downlink", "tx_errors downlink (%)", "dl_cqi", "ul_mcs",
                   "ul_n_samples", "ul_buffer [bytes]", "rx_brate uplink [Mbps]",
                   "rx_pkts uplink", "rx_errors uplink (%)", "ul_rssi", "ul_sinr",
                   "phr", "sum_requested_prbs", "sum_granted_prbs", "dl_pmi",
                   "dl_ri", "ul_n", "ul_turbo_iters"]

all_kpis = fixed_kpis + selectable_kpis  # Full list of KPIs

# Dictionary to track total size and count for averaging
kpi_sizes = defaultdict(lambda: {"total_size": 0, "count": 0})

# Traverse dataset directory and process all CSV files
for method_dir in os.listdir(base_dir):
    method_path = os.path.join(base_dir, method_dir)
    if os.path.isdir(method_path):
        for user_dir in os.listdir(method_path):
            if user_dir.startswith("users"):
                user_path = os.path.join(method_path, user_dir)
                if os.path.isdir(user_path):
                    for trial_dir in os.listdir(user_path):
                        trial_path = os.path.join(user_path, trial_dir)
                        if trial_dir.startswith("Trial_") and os.path.isdir(trial_path):
                            for file in os.listdir(trial_path):
                                if file.endswith(".csv"):  # Process only CSV files
                                    file_path = os.path.join(trial_path, file)
                                    try:
                                        df = pd.read_csv(file_path, dtype=str)  # Read CSV as strings to measure size
                                        for kpi in all_kpis:
                                            if kpi in df.columns:
                                                column_size = df[kpi].astype(str).apply(len).sum()
                                                kpi_sizes[kpi]["total_size"] += column_size
                                                kpi_sizes[kpi]["count"] += len(df)
                                    except Exception as e:
                                        print(f"Error processing {file_path}: {e}")

# Compute statistics for each KPI
kpi_stats = {}
for kpi in all_kpis:
    if kpi_sizes[kpi]["count"] > 0:
        sizes = df[kpi].astype(str).apply(len).tolist()  # Get list of sizes for the KPI
        min_size = np.min(sizes)
        avg_size = np.mean(sizes)
        max_size = np.max(sizes)
        std_dev_size = np.std(sizes, ddof=1)  # Sample standard deviation
    else:
        min_size = avg_size = max_size = std_dev_size = 0

    kpi_stats[kpi] = {
        "min": min_size,
        "mean": avg_size,
        "max": max_size,
        "std_dev": std_dev_size
    }

# Convert to DataFrame for saving
df_fixed = pd.DataFrame([(kpi, kpi_stats[kpi]["min"], kpi_stats[kpi]["mean"],
                          kpi_stats[kpi]["max"], kpi_stats[kpi]["std_dev"])
                         for kpi in fixed_kpis],
                        columns=["KPI", "Min Size", "Mean Size", "Max Size", "Std Dev"])

df_selectable = pd.DataFrame([(kpi, kpi_stats[kpi]["min"], kpi_stats[kpi]["mean"],
                               kpi_stats[kpi]["max"], kpi_stats[kpi]["std_dev"])
                              for kpi in selectable_kpis],
                             columns=["KPI", "Min Size", "Mean Size", "Max Size", "Std Dev"])

# Save to CSV
df_fixed.to_csv("fixed_kpi_stats.csv", index=False)
df_selectable.to_csv("selectable_kpi_stats.csv", index=False)

print("Results saved as 'fixed_kpi_stats.csv' and 'selectable_kpi_stats.csv'")
