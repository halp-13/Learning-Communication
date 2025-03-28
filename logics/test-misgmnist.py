# Description: Script de test pour le modèle MIS-GMNIST.(misGmnist.py)
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
import networkx as nx

def load_module_from_file(file_path, module_name):
    """
    Charger dynamiquement un module Python à partir d'un fichier.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def ensure_directory(directory):
    """
    Créer un répertoire s'il n'existe pas déjà.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def visualize_graph_structure(graph_structure, binary_image, failed_nodes, mis_mask=None, save_path=None):
    """
    Visualiser la structure du graphe:
    1. Après la déconnexion des nœuds
    2. Après l'algorithme MIS (optionnel)
    
    Paramètres:
    - graph_structure: Dictionnaire contenant les positions et la liste d'adjacence
    - binary_image: Tableau binaire représentant l'image originale
    - failed_nodes: Ensemble d'indices représentant les nœuds déconnectés
    - mis_mask: Tableau binaire représentant le résultat MIS (optionnel)
    - save_path: Chemin pour sauvegarder la visualisation (optionnel)
    """
    # Créer une figure avec 1 ou 2 sous-graphiques selon que mis_mask est fourni ou non
    num_plots = 2 if mis_mask is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(15, 7))
    
    if num_plots == 1:
        axes = [axes]  # Rendre axes itérable pour un indexage cohérent

    positions = graph_structure['positions']
    adj_list = graph_structure['adj_list']
    
    # Créer un graphe NetworkX pour la visualisation
    G_after_disconnect = nx.Graph()
    
    # Ajouter les nœuds
    for i in range(len(positions)):
        # Ne pas ajouter les nœuds défaillants
        if i not in failed_nodes:
            G_after_disconnect.add_node(i)
    
    # Ajouter les arêtes
    for node, neighbors in adj_list.items():
        if node not in failed_nodes:
            for neighbor in neighbors:
                if neighbor not in failed_nodes:
                    G_after_disconnect.add_edge(node, neighbor)
    
    # Définir les positions pour la visualisation
    pos = {i: (positions[i][0], -positions[i][1]) for i in G_after_disconnect.nodes()}  # Inverser y pour une meilleure visualisation
    
    # Tracer après déconnexion
    axes[0].set_title("Graph After Node Disconnection")
    
    # Dessiner les nœuds avec une couleur basée sur la valeur de l'image binaire
    node_colors = []
    for node in G_after_disconnect.nodes():
        if binary_image[node] == 1:
            node_colors.append('blue')
        else:
            node_colors.append('lightgray')
    
    nx.draw_networkx(
        G_after_disconnect, 
        pos=pos,
        node_color=node_colors,
        node_size=80,
        with_labels=True,
        labels={node: str(node) for node in G_after_disconnect.nodes()},
        font_size=6,
        ax=axes[0]
    )
    
    # Tracer après MIS si fourni
    if mis_mask is not None:
        G_after_mis = G_after_disconnect.copy()
        
        # Tracer après MIS
        axes[1].set_title("Graph After MIS Algorithm")
        
        # Définir les couleurs des nœuds en fonction du résultat MIS
        node_colors_mis = []
        for node in G_after_mis.nodes():
            if mis_mask[node] == 1:  # Le nœud fait partie du MIS
                node_colors_mis.append('green')
            elif binary_image[node] == 1:  # Le nœud est actif mais ne fait pas partie du MIS
                node_colors_mis.append('blue')
            else:  # Le nœud est inactif
                node_colors_mis.append('lightgray')
        
        nx.draw_networkx(
            G_after_mis, 
            pos=pos,
            node_color=node_colors_mis,
            node_size=80,
            with_labels=True,
            labels={node: str(node) for node in G_after_mis.nodes()},
            font_size=6,
            ax=axes[1]
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph visualization saved to {save_path}")
    
    plt.show()
    plt.close(fig)
    
    return fig

# Définition de l'architecture de l'auto-encodeur CNN
class CNNAutoencoder(nn.Module):
    '''
    Auto-encodeur CNN pour les images MNIST avec des graphes géométriques et des pixels déconnectés.
    Cette version correspond à l'architecture du fichier modèle sauvegardé.
    '''
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Mis à jour pour correspondre à l'architecture du modèle sauvegardé
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, failure_mask=None):
        x = self.encoder(x)
        x = self.decoder(x)
        
        if failure_mask is not None:
            x = x * failure_mask
            
        return x

def check_and_train_mis_gmnist_model(disconnect_percentage=0.6, epochs=15):
    """
    Vérifier si le modèle MIS-GMNIST est déjà entraîné. Sinon, appeler
    le script 'misGmnist.py' pour l'entraîner. Retourne True si le modèle
    est disponible, False sinon.
    """
    model_filename = f"mis-gmodel-mnist-{int(disconnect_percentage * 100)}.pth"
    processed_train_file = "processed_train_data.pkl"
    processed_test_file = "processed_test_data.pkl"

    # Déterminer si l'entraînement est nécessaire
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
            # Entraîner en appelant le script misGmnist.py
            subprocess.run(["python", "misGmnist.py"], check=True)
            print("Training completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during model training (script call failed): {e}")
            return False
    else:
        print("Model and processed data found. Skipping training step.")

    # Vérification finale si le fichier modèle existe maintenant
    if not os.path.exists(model_filename):
        print("Error: Model file still not found after training attempt.")
        return False
    
    return True

def run_mis_gmnist_test(disconnect_percentage=0.6, num_epochs=15, num_samples_to_save=10):
    """
    Exécuter la procédure de test pour le modèle MIS-GMNIST entraîné.
    Charge le modèle, construit le jeu de données de test et sauvegarde
    les échantillons reconstruits avec les visualisations de graphes.
    """
    print("\n=== Starting MIS-GMNIST Test ===")
    print(f"Disconnect Percentage: {disconnect_percentage*100}%")
    print(f"Number of Epochs (trained): {num_epochs}")
    print(f"Number of Samples to Save: {num_samples_to_save}")

    # Charger dynamiquement le module MIS
    module_path = "misGmnist.py"
    module_name = "misGmnist"
    
    if not os.path.exists(module_path):
        print(f"Error: '{module_path}' not found. Cannot load MIS-GMNIST module.")
        return
    
    mis_module = load_module_from_file(module_path, module_name)

    # Extraire les attributs/fonctions nécessaires de misGmnist.py
    create_geometric_graph_structure = mis_module.create_geometric_graph_structure
    create_binary_image = mis_module.create_binary_image
    apply_node_failures = mis_module.apply_node_failures
    compute_mis = mis_module.compute_mis
    device = mis_module.device
    BATCH_SIZE = mis_module.BATCH_SIZE  

    # Nom du fichier modèle basé sur le DISCONNECT_PERCENTAGE choisi
    model_filename = f"mis-gmodel-mnist-{int(disconnect_percentage * 100)}.pth"

    # Répertoire des résultats
    results_folder = "misgmnist-results"
    ensure_directory(results_folder)
    graph_folder = os.path.join(results_folder, "graph_visualizations")
    ensure_directory(graph_folder)

    # Charger le modèle entraîné - Utilisation de notre classe CNNAutoencoder fixe
    print("\nLoading trained model...")
    model = CNNAutoencoder().to(device)
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()

    # Charger le jeu de données de test
    print("Loading test dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # Créer la structure du graphe
    graph_structure = create_geometric_graph_structure()
    
    # Traiter et sauvegarder les échantillons
    print(f"Saving {num_samples_to_save} reconstructed samples with graph visualizations...")
    count_saved = 0

    for idx in range(len(test_dataset)):
        if count_saved >= num_samples_to_save:
            break
            
        # Obtenir l'image et créer une version binaire
        image, label = test_dataset[idx]
        binary_image = create_binary_image(image)
        
        # Appliquer les défaillances de nœuds et calculer MIS
        failed_nodes = apply_node_failures(graph_structure)
        mis_mask = compute_mis(binary_image, graph_structure, failed_nodes)
        
        # Créer un masque de défaillance
        failure_mask = np.ones(graph_structure['n_nodes'])
        for node in failed_nodes:
            failure_mask[node] = 0
            
        # Créer une image masquée
        masked_image = binary_image * mis_mask * failure_mask
        
        # Convertir en tenseurs pour l'entrée du modèle
        masked_tensor = torch.FloatTensor(masked_image).view(1, 1, 28, 28).to(device)
        original_tensor = torch.FloatTensor(binary_image).view(1, 1, 28, 28).to(device)
        failure_tensor = torch.FloatTensor(failure_mask).view(1, 1, 28, 28).to(device)
        
        # Générer la reconstruction
        with torch.no_grad():
            reconstructed = model(masked_tensor)  # Le modèle inclut maintenant une couche sigmoid
        
        # Convertir les tenseurs en tableaux NumPy pour le tracé
        masked_img_np = masked_tensor.squeeze().cpu().numpy()
        original_img_np = original_tensor.squeeze().cpu().numpy()
        failure_mask_np = failure_tensor.squeeze().cpu().numpy()
        reconstructed_np = reconstructed.squeeze().cpu().numpy()

        # Tracer et sauvegarder les images
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
        
        sample_filename = f"sample_{count_saved}_label_{label}.png"
        save_path = os.path.join(results_folder, sample_filename)
        plt.savefig(save_path)
        plt.close(fig)
        
        # Générer et sauvegarder la visualisation du graphe
        graph_filename = f"graph_{count_saved}_label_{label}.png"
        graph_save_path = os.path.join(graph_folder, graph_filename)
        
        visualize_graph_structure(
            graph_structure=graph_structure,
            binary_image=binary_image,
            failed_nodes=failed_nodes,
            mis_mask=mis_mask,
            save_path=graph_save_path
        )
        
        print(f"Saved sample {count_saved} (label {label}) with graph visualization")
        count_saved += 1

    print(f"\nSaved {count_saved} samples with graph visualizations to {results_folder}")
    print("Graph visualizations are in the 'graph_visualizations' subfolder")

def main():
    """
    Fonction principale pour orchestrer l'étape de vérification/entraînement
    puis exécuter la procédure de test MIS-GMNIST.
    """
    # Vous pouvez facilement personnaliser ces paramètres selon vos besoins.
    DISCONNECT_PERCENTAGE = 0.6
    NUM_EPOCHS = 15
    NUM_SAMPLES_TO_SAVE = 10

    print("\n=== MIS-GMNIST Test Script ===")
    print(f"DISCONNECT_PERCENTAGE: {DISCONNECT_PERCENTAGE}")
    print(f"NUM_EPOCHS (if training needed): {NUM_EPOCHS}")
    print(f"NUM_SAMPLES_TO_SAVE: {NUM_SAMPLES_TO_SAVE}")
    print("================================\n")

    # 1) Vérifier et entraîner le modèle si nécessaire
    model_ok = check_and_train_mis_gmnist_model(
        disconnect_percentage=DISCONNECT_PERCENTAGE,
        epochs=NUM_EPOCHS
    )
    if not model_ok:
        print("Could not verify or train the model. Exiting.")
        sys.exit(1)
    
    # 2) Exécuter la procédure de test (reconstruction et sauvegarde d'échantillons)
    run_mis_gmnist_test(
        disconnect_percentage=DISCONNECT_PERCENTAGE,
        num_epochs=NUM_EPOCHS,
        num_samples_to_save=NUM_SAMPLES_TO_SAVE
    )

    print("\nTest script completed successfully!")

if __name__ == "__main__":
    main()