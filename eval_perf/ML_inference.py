import os
import re
import numpy as np
import matplotlib.pyplot as plt
import csv


def parse_log_file(file_path):
    """Parse the log file to extract inference times and memory usage."""
    inference_times = []
    memory_usages = []

    with open(file_path, 'r') as file:
        for line in file:
            # Extract inference times
            match_inference = re.search(r"Inference completed in ([0-9.]+) seconds", line)
            if match_inference:
                inference_times.append(float(match_inference.group(1)))

            # Extract memory usage
            match_memory = re.search(r"Total memory usage of UE data \(imsi=[0-9]+\): ([0-9]+) bytes", line)
            if match_memory:
                memory_usages.append(int(match_memory.group(1)))

    return inference_times, memory_usages


def analyze_and_plot(data, title, unit, save_dir, plot_filename):
    """Analyze the data, plot a histogram, and save the plot."""
    data = np.array(data)
    stats = {
        "Metric": title,
        "Unit": unit,
        "Min": np.min(data),
        "Max": np.max(data),
        "Mean": np.mean(data),
        "Std Dev": np.std(data),
    }

    # Print statistics
    print(f"{title}:")
    for key, value in stats.items():
        if key not in {"Metric", "Unit"}:
            print(f"  {key}: {value:.2f} {unit}")
    print()

    # Plot histogram
    plt.hist(data, bins=20, edgecolor='black', alpha=0.75)
    plt.title(f"{title} Histogram")
    plt.xlabel(f"{title} ({unit})")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir, plot_filename))  # Save the plot
    plt.close()  # Close the plot to free up memory

    return stats


def save_statistics_to_csv(statistics, save_dir, filename):
    """Save the statistics to a CSV file."""
    csv_path = os.path.join(save_dir, filename)
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["Metric", "Unit", "Min", "Max", "Mean", "Std Dev"])
        writer.writeheader()
        writer.writerows(statistics)
    print(f"Statistics saved to {csv_path}")


def main():
    root_dir = "expert_config_inference"
    save_dir = "analysis_results"
    os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

    inference_times = []
    memory_usages = []

    # Walk through all subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "xapp-logger.log":
                file_path = os.path.join(subdir, file)
                print(f"Parsing file: {file_path}")
                times, usages = parse_log_file(file_path)
                inference_times.extend(times)
                memory_usages.extend(usages)

    statistics = []

    # Analyze and plot inference times
    if inference_times:
        stats = analyze_and_plot(
            inference_times,
            "Inference Time",
            "seconds",
            save_dir,
            "inference_time_histogram.png",
        )
        statistics.append(stats)
    else:
        print("No inference times found.")

    # Analyze and plot memory usage
    if memory_usages:
        stats = analyze_and_plot(
            memory_usages,
            "Memory Usage",
            "bytes",
            save_dir,
            "memory_usage_histogram.png",
        )
        statistics.append(stats)
    else:
        print("No memory usage data found.")

    # Save statistics to CSV
    if statistics:
        save_statistics_to_csv(statistics, save_dir, "statistics_summary.csv")


if __name__ == "__main__":
    main()
