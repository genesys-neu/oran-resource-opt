import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from ORAN_models import TransformerNN

np.random.seed(29)


class ORANDataset(Dataset):
    def __init__(self, data, labels, experiment_ids, sequence_length, max_attempts=16):
        self.data = data
        self.labels = labels
        self.experiment_ids = experiment_ids  # New: pass the ExperimentID
        self.sequence_length = sequence_length
        self.max_attempts = max_attempts
        self.valid_indices = self._generate_valid_indices()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Get the actual index from the valid indices list
        actual_idx = self.valid_indices[idx]

        # Initialize slice
        end_idx = actual_idx + self.sequence_length
        sample = self.data.iloc[actual_idx:end_idx].values
        label_tensor = self.labels.iloc[actual_idx:end_idx].values

        # Return the sample and its corresponding label
        label = torch.tensor(label_tensor[0], dtype=torch.long)  # Use the first label
        sample_tensor = torch.tensor(sample, dtype=torch.float32)
        return sample_tensor, label

    def _generate_valid_indices(self):
        valid_indices = []
        for idx in range(len(self.data) - self.sequence_length + 1):
            sample = self.data.iloc[idx:idx + self.sequence_length].values
            label_tensor = self.labels.iloc[idx:idx + self.sequence_length].values
            experiment_ids = self.experiment_ids.iloc[idx:idx + self.sequence_length].values  # Get ExperimentIDs
            attempts = 0
            while len(set(label_tensor)) > 1 and attempts < self.max_attempts:
                if idx > 0:
                    idx -= 1
                    sample = self.data.iloc[idx:idx + self.sequence_length].values
                    label_tensor = self.labels.iloc[idx:idx + self.sequence_length].values
                    experiment_ids = self.experiment_ids.iloc[idx:idx + self.sequence_length].values
                    attempts += 1
                else:
                    break

            # Only add to valid indices if labels are consistent and ExperimentIDs match
            if len(set(label_tensor)) == 1 and len(set(experiment_ids)) == 1:  # Labels and ExperimentID are consistent
                valid_indices.append(idx)

        return valid_indices


# Load feature statistics for normalization
def load_feature_stats(stats_file):
    feature_stats = pd.read_csv(stats_file)
    # Create dictionaries for Min and Max values
    min_values = feature_stats.set_index('Feature')['Min'].to_dict()
    max_values = feature_stats.set_index('Feature')['Max'].to_dict()
    # Return both min and max values
    return min_values, max_values


# Normalize the data using the provided max values
def normalize_data(data, min_values, max_values):
    # Normalize the data using the provided min and max values
    for column in min_values:
        min_val = min_values[column]
        max_val = max_values[column]

        if max_val != min_val:  # Avoid division by zero
            # Apply min-max normalization
            data[column] = (data[column] - min_val) / (max_val - min_val)
        else:
            # If min == max, set the value to 0 (or handle as needed)
            data[column] = 0  # Alternatively, you could leave it as is or apply other logic
    return data


# Update the load_data function to include normalization
def load_data(file, chunk, min_values, max_values):
    # Load the full CSV into memory (if possible)
    data = pd.read_csv(file)

    # Randomly select indices for validation/testing (10% of data)
    total_length = len(data)
    num_chunks = total_length // chunk
    valid_test_indices = np.random.choice(num_chunks, size=int(num_chunks * 0.1), replace=False)

    # Normalize the data using min-max normalization
    data = normalize_data(data, min_values, max_values)

    # Initialize empty DataFrames for validation and testing data
    validation_data = pd.DataFrame()
    testing_data = pd.DataFrame()

    # Iterate over each index and extract the corresponding chunk
    for idx in valid_test_indices:
        start = idx * chunk
        end = (idx + 1) * chunk
        validation_data = pd.concat([validation_data, data.iloc[start:end]])

    # Copy validation_data to testing_data (assuming they are identical)
    testing_data = validation_data.copy()

    # Remove validation/testing rows from the training data
    training_data = data.drop(validation_data.index)

    # Extract labels and ExperimentIDs
    training_labels = training_data.pop('Label')
    validation_labels = validation_data.pop('Label')
    testing_labels = testing_data.pop('Label')

    training_experiment_ids = training_data.pop('ExperimentID')
    validation_experiment_ids = validation_data.pop('ExperimentID')
    testing_experiment_ids = testing_data.pop('ExperimentID')

    return training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels, \
           training_experiment_ids, validation_experiment_ids, testing_experiment_ids


# def init_weights(m):
#     if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
#         nn.init.xavier_uniform_(m.weight)  # Apply Xavier initialization
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)  # Initialize biases to 0


# Main training script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_input", default='final_dataset.csv',
                        help="file containing all the training data")
    parser.add_argument("-t", "--trial_version", default='',
                        help="add identifier for this trial")
    parser.add_argument("-s", "--slice_length", type=int, default=32, help="Slice length for the Transformer")
    parser.add_argument("-nh", "--num_heads", type=int, default=1, help="Number of Transformer heads")
    parser.add_argument("-p", "--pretrained_model", default=None,
                        help="Path to the pre-trained model weights (.pth file)")
    parser.add_argument("--feature_stats", default='feature_stats.csv',
                        help="File containing feature min and max values for normalization")

    args = parser.parse_args()
    input_file = args.file_input
    t_v = args.trial_version
    slice_length = args.slice_length
    num_heads = args.num_heads
    pretrained_model = args.pretrained_model
    feature_stats_file = args.feature_stats
    learning_rate = 0.001

    chunk_size = 1024
    training_metrics = {'epochs': [], 'training_loss': [], 'training_accuracy': [], 'validation_loss': [],
                        'validation_accuracy': [], 'confusion_matrix': []}

    # Load feature statistics for normalization
    min_values, max_values = load_feature_stats(feature_stats_file)

    # Load data using the updated load_data function
    train_data, validate_data, test_data, train_label, validate_label, test_label, \
    train_experiment_ids, validate_experiment_ids, test_experiment_ids = load_data(input_file, chunk_size, min_values,
                                                                                   max_values)

    # Check the dimensions of the loaded data (optional, for debugging)
    # print(f"Training Data Shape: {train_data.shape}")
    # print(f"Training Labels Shape: {train_label.shape}")
    # print(f"Validation Data Shape: {validate_data.shape}")
    # print(f"Validation Labels Shape: {validate_label.shape}")
    # print(f"Test Data Shape: {test_data.shape}")
    # print(f"Test Labels Shape: {test_label.shape}")

    # Create dataset instances using the updated ORANDataset class
    train_dataset = ORANDataset(train_data, train_label, train_experiment_ids, slice_length)
    val_dataset = ORANDataset(validate_data, validate_label, validate_experiment_ids, slice_length)
    test_dataset = ORANDataset(test_data, test_label, test_experiment_ids, slice_length)

    # Check the lengths of the dataset instances
    # print(f"Train Dataset Length: {len(train_dataset)}")
    # print(f"Validation Dataset Length: {len(val_dataset)}")
    # print(f"Test Dataset Length: {len(test_dataset)}")

    # Check one sample from each dataset to verify dimensions
    # sample_train, label_train = train_dataset[0]
    # print(f"Sample Training Data Shape: {sample_train.shape}, Label: {label_train}")
    # sample_val, label_val = val_dataset[0]
    # print(f"Sample Validation Data Shape: {sample_val.shape}, Label: {label_val}")
    # sample_test, label_test = test_dataset[0]
    # print(f"Sample Test Data Shape: {sample_test.shape}, Label: {label_test}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define Transformer model
    print(f'Using Transformer with {num_heads} heads and slice size {slice_length}')
    model = TransformerNN(slice_len=slice_length, nhead=num_heads).to(device)
    # model.apply(init_weights)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f'{name}: mean={param.mean().item()}, std={param.std().item()}')

    if pretrained_model:
        try:
            model.load_state_dict(torch.load(pretrained_model))
            print(f"Loaded pre-trained model from {pretrained_model}")
            learning_rate = 0.0001
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Training Transformer model from scratch instead.")
    else:
        print("Training Transformer model from scratch")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')
    best_val_accuracy = 0
    patience = 10
    patience_counter = 0

    # Save the initial state of the model
    torch.save(model.state_dict(), f'best_model_{t_v}.pth')

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        batch_size = 1000
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training phase
        model.train()

        running_loss = 0.0
        running_accuracy = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            if inputs is None or labels is None:
                continue  # Skip this batch if it contains None values

            # Check for NaN values in inputs and labels
            if torch.isnan(inputs).any():
                print("NaN detected in inputs!")

            # Check for Inf values in inputs and labels
            if torch.isinf(inputs).any():
                print("Inf detected in inputs!")

            # Log the shape of inputs and labels for debugging
            # print(f"Inputs Shape: {inputs.shape}")
            # print(f"Labels Shape: {labels.shape}")

            # Get basic statistics for the inputs to understand their scale and range
            # if inputs.isfinite().all():  # Ensure we don't get an error on invalid values
            #     print(f"Inputs Statistics - Mean: {inputs.mean().item()}, "
            #           f"Min: {inputs.min().item()}, "
            #           f"Max: {inputs.max().item()}")

            # Ensure the inputs are of the correct type
            # print(f"Inputs Data Type: {inputs.dtype}")
            # print(f"Labels Data Type: {labels.dtype}")  # Print data type of labels to check
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)  # shape [batch_size, num_classes]

            # Print the size of outputs and labels for verification
            # print(f"Outputs size: {outputs.size()}")
            # print(f"Labels size: {labels.size()}")

            # Check for NaN in model output
            # if torch.isnan(outputs).any():
            #     print("NaN detected in model output")
            #     # Remove rows with NaN outputs (if NaN is found)
            #     valid_idx = ~torch.isnan(outputs).any(dim=1)
            #     inputs = inputs[valid_idx]
            #     labels = labels[valid_idx]
            #     outputs = outputs[valid_idx]
            #     # print(f"Cleaned output size: {outputs.size()}")
            #     # print(f"Cleaned labels size: {labels.size()}")
            #     if len(outputs) == 0:
            #         continue  # If after removal no valid rows remain, skip this batch
            # else:
            #     # Check if outputs are within the expected range
            #     if torch.isinf(outputs).any():
            #         print("Inf detected in model output!")
            #     # Print statistics about outputs for further analysis
            #     print(f"Outputs Statistics - Mean: {outputs.mean().item()}, "
            #           f"Min: {outputs.min().item()}, Max: {outputs.max().item()}")
            #     print(f"Outputs Shape: {outputs.shape}")

            # Calculate loss
            loss = criterion(outputs, labels)  # labels should be [batch_size] with class indices
            # print(f"Loss: {loss.item()}")

            # Check for NaN in the loss
            # if torch.isnan(loss).any():
            #     print("NaN detected in loss!")
            #     continue  # Skip this batch if NaN is found

            loss.backward()
            # Mask gradients with NaN
            # mask_count = 0
            # masked_params_count = 0  # To count the number of parameters with NaNs
            # for param in model.parameters():
            #     if param.grad is not None:
            #         # Check for NaN values in the gradient
            #         nan_mask = torch.isnan(param.grad)
            #         if nan_mask.any():
            #             # Replace NaNs with 0s
            #             param.grad[nan_mask] = 0.0
            #             masked_params_count += 1  # Increment count for this parameter
            #             mask_count += nan_mask.sum().item()  # Sum of NaNs in this parameter
            # if masked_params_count > 0:
            #     print(f'NaN detected in gradient, masked {masked_params_count} '
            #           f'parameters with a total of {mask_count} NaNs.')

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # Clip gradients
            optimizer.step()

            # Predictions and accuracy
            _, predicted = torch.max(outputs, dim=1)  # Get predicted class indices
            running_accuracy += (predicted == labels).sum().item() / labels.size(0)

            # Accumulate the loss
            running_loss += loss.item()

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():  # Disable gradient calculation during validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels)

                # Predictions and accuracy
                _, predicted = torch.max(outputs, dim=1)
                val_accuracy += (predicted == labels).sum().item()  # Accumulate the correct predictions
                val_loss += loss.item()  # Accumulate the loss

        # Normalize the accumulated accuracy and loss by the number of batches
        val_accuracy /= len(val_loader.dataset)  # Divide by the total number of samples in the validation set
        val_loss /= len(val_loader)  # Divide by the number of batches

        scheduler.step()  # Step the learning rate scheduler

        # Early stopping and saving the best model based on validation loss or accuracy
        if val_loss < best_val_loss or val_accuracy > best_val_accuracy:
            torch.save(model.state_dict(), f'best_model_{t_v}.pth')  # Save model
            patience_counter = 0  # Reset patience counter since we have improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
        else:
            patience_counter += 1  # Increment patience counter if no improvement

        # Print training and validation metrics
        print(f'Epoch {epoch + 1}, Patience Counter: {patience_counter}, '
              f'Training Loss: {running_loss / len(train_loader):.4f}, '
              f'Training Accuracy: {running_accuracy / len(train_loader):.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

        # Store metrics for analysis
        training_metrics['epochs'].append(epoch + 1)
        training_metrics['training_loss'].append(running_loss / len(train_loader))
        training_metrics['training_accuracy'].append(running_accuracy / len(train_loader))
        training_metrics['validation_loss'].append(val_loss)
        training_metrics['validation_accuracy'].append(val_accuracy)

        # Early stopping based on the patience counter
        if patience_counter >= patience:
            print(f'Validation loss has not improved for {patience} epochs. Stopping training.')
            break  # Stop training if no improvement in validation loss/accuracy

    # Testing phase
    model.load_state_dict(torch.load(f'best_model_{t_v}.pth'))
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_predictions = []
    test_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)  # Get predicted class indices
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(test_targets, test_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)

    training_metrics['confusion_matrix'].append(conf_matrix.tolist())

    with open(f'training_log_{t_v}.json', 'w') as jsonfile:
        json.dump(training_metrics, jsonfile)
