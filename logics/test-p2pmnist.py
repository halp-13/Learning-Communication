import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import importlib.util
import subprocess

def load_module_from_file(file_path, module_name):
    """Dynamically load a Python module from a file."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def ensure_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def check_and_train_model(disconnect_percentage, buffer_size, epochs=15, learning_rate=1e-4):
    """Check if model exists with the specified disconnect percentage, train if not."""
    model_filename = f"autoencoder_model_P2P_mnist_disconnect_{disconnect_percentage}pct.pth"
    
    if os.path.exists(model_filename):
        print(f"Found existing model: {model_filename}")
        return True
    
    print(f"Model {model_filename} not found. Training new model...")
    
    # Check if the training script exists
    if os.path.exists("p2pmnist.py"):
        try:
            # Try to import the training module
            p2p_module = load_module_from_file("p2pmnist.py", "p2pmnist")
            
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Set up data transformation
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x > 0.5).float()),  # Binarization
            ])
            
            # Load MNIST dataset
            train_dataset = datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transform
            )
            
            # Create DataLoader
            batch_size = 128
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Create the CNN model
            cnn_model = p2p_module.CNNAutoencoder().to(device)
            
            # Train the model
            p2p_module.train_cnn_model_with_dynamic_disconnections(
                cnn_model,
                train_loader,
                device,
                epochs=epochs,
                disconnect_percentage=disconnect_percentage,
                buffer_size=buffer_size,
                learning_rate=learning_rate
            )
            
            return True
        except Exception as e:
            print(f"Error during model training: {e}")
            return False
    else:
        print("Error: p2pmnist.py not found. Cannot train model.")
        return False

def run_p2p_mnist_test(disconnect_percentage, buffer_size, test_steps):
    """Run the P2P MNIST test with the specified parameters."""
    print("\n=== Starting P2P MNIST Test ===")
    print(f"Disconnect Percentage: {disconnect_percentage}%")
    print(f"Buffer Size: {buffer_size}")
    print(f"Test Steps: {test_steps}")
    
    # Create results directory
    results_dir = f"test_results_{disconnect_percentage}pct"
    ensure_directory(results_dir)
    
    # Load the p2pmnist module
    p2p_module = load_module_from_file("p2pmnist.py", "p2pmnist")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the trained model
    model_filename = f"autoencoder_model_P2P_mnist_disconnect_{disconnect_percentage}pct.pth"
    cnn_model = p2p_module.CNNAutoencoder().to(device)
    cnn_model.load_state_dict(torch.load(model_filename))
    cnn_model.eval()
    
    # Load the test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),  # Binarization
    ])
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Run tests
    results = {
        "overall_accuracy": [],
        "sample_results": []
    }
    
    for sample_idx in range(test_steps):
        print(f"\n--- Testing on Sample {sample_idx+1}/{test_steps} ---")
        
        # Select a random image
        idx = random.randint(0, len(test_dataset) - 1)
        image, digit = test_dataset[idx]
        
        # Each pixel (0/1) of the image represents a Bob
        image = image.to(device)
        flat_image = image.view(-1)  # (784,)
        
        # Create Alice object
        alice = p2p_module.Alice(p_one=0.5, buffer_size=buffer_size, use_cnn=True)
        alice.set_cnn_model(cnn_model)
        
        # Create 784 Bobs (one for each pixel of the MNIST image)
        bobs = []
        for i in range(784):
            # The probability for each Bob is based on the corresponding pixel
            p_one = float(flat_image[i].item())  # 0.0 or 1.0
            bob = p2p_module.Node(f"Bob_{i}", p_one, buffer_size=buffer_size)
            bobs.append(bob)
        
        # Determine which Bobs will be disconnected
        num_disconnected = int(784 * disconnect_percentage / 100)
        disconnected_bobs = random.sample(range(784), num_disconnected)
        
        # Mark these Bobs as disconnected
        for bob_id in disconnected_bobs:
            bobs[bob_id].is_disconnected = True
        
        # Create a mask representing connected (1) and disconnected (0) Bobs
        partial_flat_image = flat_image.clone()
        for bob_id in disconnected_bobs:
            partial_flat_image[bob_id] = -1

        partial_image = partial_flat_image.view(1, 1, 28, 28)
        
        # Predict the missing bits
        true_bits = []
        predicted_bits = []
        
        for bob_id in disconnected_bobs:
            true_bit = int(flat_image[bob_id].item())
            # Predict the bit using the CNN model
            predicted_bit = alice.predict_message(bob_id, partial_image)
            
            true_bits.append(true_bit)
            predicted_bits.append(predicted_bit)
        
        # Calculate accuracy
        correct_count = sum(1 for t, p in zip(true_bits, predicted_bits) if t == p)
        accuracy = (correct_count / len(disconnected_bobs)) * 100 if disconnected_bobs else 0
        
        print(f"Sample {sample_idx+1} (digit {digit}):")
        print(f"  - Disconnected Bobs: {num_disconnected}/{784} ({disconnect_percentage}%)")
        print(f"  - Correctly predicted: {correct_count}/{len(disconnected_bobs)}")
        print(f"  - Accuracy: {accuracy:.2f}%")
        
        # Record the results
        results["overall_accuracy"].append(accuracy)
        results["sample_results"].append({
            "sample_idx": sample_idx,
            "digit": digit.item() if isinstance(digit, torch.Tensor) else digit,
            "disconnected_bobs": disconnected_bobs,
            "true_bits": true_bits,
            "predicted_bits": predicted_bits,
            "accuracy": accuracy
        })
        
        # Visualize and save the original, partial, and reconstructed images
        with torch.no_grad():
            reconstructed = cnn_model(partial_image)
            reconstructed = torch.sigmoid(reconstructed)
        
        # Convert to numpy for display
        orig_img = image.cpu().view(28, 28).numpy()
        partial_img = partial_image.cpu().view(28, 28).numpy()
        recon_img = reconstructed.cpu().view(28, 28).numpy()
        
        # Save the images
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(orig_img, cmap='gray')
        plt.title(f"Original (Digit {digit})")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(partial_img, cmap='gray')
        plt.title(f"Partial ({disconnect_percentage}% Disconnected)")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(recon_img, cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/sample_{sample_idx+1}.png")
        plt.close()
    
    # Final report
    avg_accuracy = sum(results["overall_accuracy"]) / len(results["overall_accuracy"])
    print(f"\n--- Final Results over {test_steps} samples ---")
    print(f"Average prediction accuracy: {avg_accuracy:.2f}%")
    
    # Save accuracy distribution chart
    accuracies = [result["accuracy"] for result in results["sample_results"]]
    
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribution of Prediction Accuracies')
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.axvline(avg_accuracy, color='red', linestyle='dashed', 
                linewidth=2, label=f'Mean: {avg_accuracy:.2f}%')
    plt.legend()
    plt.savefig(f"{results_dir}/accuracy_distribution.png")
    plt.close()
    
    # Save accuracy by digit chart
    digit_accuracies = {}
    for result in results["sample_results"]:
        digit = result["digit"]
        if digit not in digit_accuracies:
            digit_accuracies[digit] = []
        digit_accuracies[digit].append(result["accuracy"])
    
    # Average by digit
    digit_avg_acc = {digit: sum(accs)/len(accs) for digit, accs in digit_accuracies.items() if accs}
    
    # Display performance by digit
    if digit_avg_acc:  # Only if we have digit data
        digits = sorted(digit_avg_acc.keys())
        avg_accs = [digit_avg_acc[d] for d in digits]
        
        plt.figure(figsize=(10, 6))
        plt.bar(digits, avg_accs, color='lightgreen', edgecolor='black')
        plt.title('Average Prediction Accuracy by Digit')
        plt.xlabel('Digit')
        plt.ylabel('Average Accuracy (%)')
        plt.xticks(digits)
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig(f"{results_dir}/accuracy_by_digit.png")
        plt.close()
    
    print(f"Test results saved to {results_dir}/")
    return results

def main():
    # Paramètres simples - modifiez directement ces valeurs
    disconnect_percentage = 60  # Pourcentage de Bobs à déconnecter
    buffer_size = 25           # Taille du buffer
    test_steps = 50            # Nombre d'échantillons de test
    epochs = 15                # Nombre d'époques si entraînement nécessaire
    learning_rate = 1e-4       # Taux d'apprentissage si entraînement nécessaire
    
    # Afficher les paramètres utilisés
    print("\n=== P2P MNIST Test Parameters ===")
    print(f"Disconnect percentage: {disconnect_percentage}%")
    print(f"Buffer size: {buffer_size}")
    print(f"Nomber of test steps: {test_steps}")
    print(f"Nomber of epochs: {epochs}")
    print(f"Learning rate : {learning_rate}")
    print("================================\n")
    
    # Vérifier si le répertoire de données existe, le créer si nécessaire
    if not os.path.exists('./data'):
        print("data directory not found, creating...")
    
    # Vérifier et entraîner le modèle si nécessaire
    model_trained = check_and_train_model(
        disconnect_percentage, 
        buffer_size,
        epochs,
        learning_rate
    )
    
    if not model_trained:
        print(" it was not possible to train the model.")
        return
    
    # Exécuter le test
    run_p2p_mnist_test(
        disconnect_percentage,
        buffer_size,
        test_steps
    )
    
    print("Test finished with sucsess.")

if __name__ == "__main__":
    main()