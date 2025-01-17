import torch


def troubleshoot_model_loading(pretrained_model_path, model):
    """
    Troubleshoot loading of a pretrained model into the given model architecture.

    Args:
        pretrained_model_path (str): Path to the pretrained model file.
        model (torch.nn.Module): The model architecture to load the weights into.
    """
    # Load the checkpoint
    checkpoint = torch.load(pretrained_model_path, map_location="cpu")

    # Extract the model_state_dict if it exists
    if "model_state_dict" in checkpoint:
        print("Extracting 'model_state_dict' from the checkpoint...")
        state_dict = checkpoint["model_state_dict"]
    else:
        print("No 'model_state_dict' found. Assuming the checkpoint is a plain state_dict.")
        state_dict = checkpoint

    # List the keys in the state_dict
    print(f"Pretrained model keys ({len(state_dict.keys())} total):")
    print(set(state_dict.keys()))

    # List the keys in the model
    model_keys = set(model.state_dict().keys())
    print(f"\nModel parameter keys ({len(model_keys)} total):")
    print(model_keys)

    # Find missing keys (expected by the model but not in the state_dict)
    missing_keys = model_keys - set(state_dict.keys())
    if missing_keys:
        print(f"\nMissing keys in pretrained model ({len(missing_keys)}):")
        print(missing_keys)
    else:
        print("\nNo missing keys. All model parameters are present in the checkpoint.")

    # Find unexpected keys (in the state_dict but not in the model)
    extra_keys = set(state_dict.keys()) - model_keys
    if extra_keys:
        print(f"\nExtra keys in pretrained model ({len(extra_keys)}):")
        print(extra_keys)
    else:
        print("\nNo extra keys. The state_dict matches the model exactly.")

    # Try loading the state_dict with strict=False to verify compatibility
    print("\nAttempting to load state_dict with strict=False...")
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully with strict=False.")
    except Exception as e:
        print(f"Failed to load model with strict=False. Error: {e}")


# Example usage:
if __name__ == "__main__":
    # Define your model architecture
    from ORAN_models import TransformerNN  # Replace with your actual module and model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerNN(slice_len=32, nhead=1).to(device)  # Adjust params as needed

    # Path to your pretrained model
    pretrained_model_path = r"C:\Users\joshg\PycharmProjects\IMPACT\model\Transformer\trans_v1.1.32.pth"  # Replace with the actual path

    # Run the troubleshooting function
    troubleshoot_model_loading(pretrained_model_path, model)
