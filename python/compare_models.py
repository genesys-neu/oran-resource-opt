import os
import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix
from train_test import ORANDataset, load_feature_stats, load_data
from ORAN_models import TransformerNN

# Define a function to parse model parameters from the file name
def parse_model_name(model_name):
    parts = model_name.split('.')
    num_heads = int(parts[2])
    slice_length = int(parts[3])
    return num_heads, slice_length

# Define a function to load and evaluate a single model
def evaluate_model(model_path, test_dataset, device):
    model_name = os.path.basename(model_path)
    num_heads, slice_length = parse_model_name(model_name)

    model = TransformerNN(slice_len=slice_length, nhead=num_heads).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    return confusion_matrix(test_targets, test_predictions)

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(6, 3))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 24})
    plt.xlabel("Predicted Label", fontsize=18)
    plt.ylabel("True Label", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(False)

    plt.gca().spines["top"].set_visible(True)
    plt.gca().spines["right"].set_visible(True)
    plt.gca().spines["left"].set_visible(True)
    plt.gca().spines["bottom"].set_visible(True)
    plt.tight_layout()
    plt.savefig(f"conf_matrix_{model_name}.pdf", format="pdf")
    plt.close()

# Main script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare models in a directory.")
    parser.add_argument("directory", type=str, help="Directory containing model files.")
    parser.add_argument("--feature_stats", type=str, required=True, help="Path to feature statistics file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input data file.")
    parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for data loading.")
    parser.add_argument("--slice_length", type=int, required=True, help="Slice length for the dataset.")

    args = parser.parse_args()

    # Load feature statistics
    min_values, max_values = load_feature_stats(args.feature_stats)

    # Load data
    _, _, test_data, _, _, test_label, _, _, test_experiment_ids = load_data(
        args.input_file, args.chunk_size, min_values, max_values
    )

    # Create test dataset
    test_dataset = ORANDataset(test_data, test_label, test_experiment_ids, args.slice_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate all models in the directory
    model_files = [f for f in os.listdir(args.directory) if f.endswith(".pth")]
    results = {}

    for model_file in model_files:
        model_path = os.path.join(args.directory, model_file)
        print(f"Evaluating model: {model_file}")
        conf_matrix = evaluate_model(model_path, test_dataset, device)
        results[model_file] = conf_matrix.tolist()

        # Plot confusion matrix
        plot_confusion_matrix(conf_matrix, model_file)

    # Save results to JSON
    with open("model_comparison_results.json", "w") as json_file:
        json.dump(results, json_file)

    print("Comparison completed. Results saved to model_comparison_results.json.")
