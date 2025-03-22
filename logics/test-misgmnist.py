# Description: A script to test the MIS-GMNIST model. pour tester fichier misGmnist.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import importlib.util
import subprocess

def load_module_from_file(file_path, module_name):
    """
    Dynamically load a Python module from a file.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def ensure_directory(directory):
    """
    Create a directory if it doesn't exist.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def check_and_train_mis_gmnist_model(disconnect_percentage=0.6, epochs=15):
    """
    Check if the MIS-GMNIST model is already trained. If not, call
    the 'misGmnist.py' script to train it. Return True if the model
    is available, False otherwise.
    """
    model_filename = f"mis-gmodel-mnist-{int(disconnect_percentage * 100)}.pth"
    processed_train_file = "processed_train_data.pkl"
    processed_test_file = "processed_test_data.pkl"

    # Determine whether training is needed
    need_to_train = False
    if not os.path.exists(model_filename):
        need_to_train = True
    if not os.path.exists(processed_train_file) or not os.path.exists(processed_test_file):
        need_to_train = True

    if need_to_train:
        print("Model or processed data not found. Training...")
        if not os.path.exists("misGmnist.py"):
            print("Error: 'misGmnist.py' not found. Cannot train model.")
            return False
        try:
            # Train by calling the misGmnist.py script
            subprocess.run(["python", "misGmnist.py"], check=True)
            print("Training completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during model training (script call failed): {e}")
            return False
    else:
        print("Model and processed data found. Skipping training step.")

    # Final check if model file now exists
    if not os.path.exists(model_filename):
        print("Error: Model file still not found after training attempt.")
        return False
    
    return True


def run_mis_gmnist_test(disconnect_percentage=0.6, num_epochs=15, num_samples_to_save=10):
    """
    Run the testing procedure for the trained MIS-GMNIST model.
    Loads the model, constructs the test dataset, and saves a few
    reconstructed samples.
    """
    print("\n=== Starting MIS-GMNIST Test ===")
    print(f"Disconnect Percentage: {disconnect_percentage*100}%")
    print(f"Number of Epochs (trained): {num_epochs}")
    print(f"Number of Samples to Save: {num_samples_to_save}")

    # Lazy import: We'll load classes/functions from 'misGmnist.py' if needed
    # Alternatively, you can directly import if your environment is set up.
    # e.g., `from misGmnist import CNNAutoencoder, MNISTGraphDataset, ...`
    # Here we follow the dynamic loading style (like the second snippet).
    module_path = "misGmnist.py"
    module_name = "misGmnist"
    
    if not os.path.exists(module_path):
        print(f"Error: '{module_path}' not found. Cannot load MIS-GMNIST module.")
        return
    
    mis_module = load_module_from_file(module_path, module_name)

    # Extract necessary attributes/classes from misGmnist.py
    CNNAutoencoder = mis_module.CNNAutoencoder
    create_geometric_graph_structure = mis_module.create_geometric_graph_structure
    MNISTGraphDataset = mis_module.MNISTGraphDataset
    device = mis_module.device
    BATCH_SIZE = mis_module.BATCH_SIZE  # or you can define your own batch size

    # Model filename based on the chosen DISCONNECT_PERCENTAGE
    model_filename = f"mis-gmodel-mnist-{int(disconnect_percentage * 100)}.pth"

    # Results directory
    results_folder = "misresults"
    ensure_directory(results_folder)

    # Load the trained model
    print("\nLoading trained model...")
    model = CNNAutoencoder().to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()

    # Build the test dataset
    print("Loading test dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Create the graph structure and dataset
    graph_structure = create_geometric_graph_structure()
    test_graph_dataset = MNISTGraphDataset(test_dataset, graph_structure, is_train=False)
    test_graph_loader = DataLoader(test_graph_dataset, batch_size=1, shuffle=True)

    # Save a short report about the parameters used
    report_filename = os.path.join(results_folder, "report.txt")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(f"DISCONNECT_PERCENTAGE: {disconnect_percentage}\n")
        f.write(f"NUM_EPOCHS: {num_epochs}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")

    print(f"Saving {num_samples_to_save} reconstructed samples...")
    count_saved = 0

    for idx, (masked_img, original_img, label, failure_mask) in enumerate(test_graph_loader):
        if count_saved >= num_samples_to_save:
            break
        
        masked_img = masked_img.to(device)
        original_img = original_img.to(device)
        failure_mask = failure_mask.to(device)
        
        with torch.no_grad():
            output_logits = model(masked_img)      # raw logits
            reconstructed = torch.sigmoid(output_logits)  # convert to [0, 1])
        
        # Convert tensors to NumPy arrays for plotting
        masked_img_np = masked_img.squeeze().cpu().numpy()
        original_img_np = original_img.squeeze().cpu().numpy()
        failure_mask_np = failure_mask.squeeze().cpu().numpy()
        reconstructed_np = reconstructed.squeeze().cpu().numpy()

        # Plotting and saving the images
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        
        axes[0].imshow(original_img_np, cmap='gray')
        axes[0].set_title("Original")
        axes[0].axis('off')

        axes[1].imshow(failure_mask_np * original_img_np, cmap='gray')
        axes[1].set_title("After Drop")
        axes[1].axis('off')

        axes[2].imshow(masked_img_np, cmap='gray')
        axes[2].set_title("After MIS")
        axes[2].axis('off')

        axes[3].imshow(reconstructed_np, cmap='gray')
        axes[3].set_title("Reconstructed")
        axes[3].axis('off')
        
        plt.tight_layout()
        
        sample_filename = f"sample_{count_saved}_label_{label.item()}.png"
        save_path = os.path.join(results_folder, sample_filename)
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"Saved sample {count_saved} to {save_path}")
        count_saved += 1


def main():
    """
    Main function to orchestrate the check/train step
    and then run the MIS-GMNIST test procedure.
    """
    # You can easily customize these parameters as needed.
    DISCONNECT_PERCENTAGE = 0.6
    NUM_EPOCHS = 15
    NUM_SAMPLES_TO_SAVE = 10

    print("\n=== MIS-GMNIST Test Script ===")
    print(f"DISCONNECT_PERCENTAGE: {DISCONNECT_PERCENTAGE}")
    print(f"NUM_EPOCHS (if training needed): {NUM_EPOCHS}")
    print(f"NUM_SAMPLES_TO_SAVE: {NUM_SAMPLES_TO_SAVE}")
    print("================================\n")

    # 1) Check and train the model if necessary
    model_ok = check_and_train_mis_gmnist_model(
        disconnect_percentage=DISCONNECT_PERCENTAGE,
        epochs=NUM_EPOCHS
    )
    if not model_ok:
        print("Could not verify or train the model. Exiting.")
        sys.exit(1)
    
    # 2) Run the test procedure (reconstruction and saving samples)
    run_mis_gmnist_test(
        disconnect_percentage=DISCONNECT_PERCENTAGE,
        num_epochs=NUM_EPOCHS,
        num_samples_to_save=NUM_SAMPLES_TO_SAVE
    )

    print("\nTest script completed successfully!")

# Standard Python entry point
if __name__ == "__main__":
    main()
