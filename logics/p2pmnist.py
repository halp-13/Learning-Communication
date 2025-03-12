import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------------------------------------------------------------
#                        PARTIE 1: DÉFINITIONS DES CLASSES P2P
# -----------------------------------------------------------------------------

class Node:
    """
    Cette classe représente un nœud dans la communication P2P.
    Chaque nœud peut envoyer et recevoir des bits (0 ou 1).
    Il peut également être déconnecté.
    """
    def __init__(self, name, p_one, buffer_size=20):
        """
        name: Nom du nœud (ex: 'Alice' ou 'Bob_1')
        p_one: Probabilité qu'un message envoyé par ce nœud soit un bit 1
        buffer_size: Taille du buffer qui stocke les bits reçus
        """
        self.name = name
        self.p_one = p_one
        self.buffer_size = buffer_size
        self.is_disconnected = False
        self.received_bits = []
        self.count_ones = 0
        self.count_total = 0
    
    def send_message(self):
        if self.is_disconnected:
            return None
        bit = 1 if random.random() < self.p_one else 0
        return bit
    
    def receive_message(self, bit):
        # Si le buffer est plein, retirez le bit le plus ancien et ajustez les statistiques
        if len(self.received_bits) >= self.buffer_size:
            oldest_bit = self.received_bits.pop(0)  # Retire le plus ancien bit
            if oldest_bit == 1:
                self.count_ones -= 1
            self.count_total -= 1
        
        self.received_bits.append(bit)
        self.count_total += 1
        if bit == 1:
            self.count_ones += 1
    
    def guess_message(self):
        """
        Devine un bit lorsque l'autre nœud est déconnecté.
        """
        if self.count_total == 0:
            return 0
        
        # Utilisation du buffer pour le calcul de probabilité
        prob_one = self.count_ones / self.buffer_size if self.count_total >= self.buffer_size else self.count_ones / self.count_total
        bit_guess = 1 if random.random() < prob_one else 0
        return bit_guess

    def to_dict(self):
        return {
            "name": self.name,
            "p_one": self.p_one,
            "buffer_size": self.buffer_size,
            "is_disconnected": self.is_disconnected,
            "received_bits": self.received_bits,
            "count_ones": self.count_ones,
            "count_total": self.count_total
        }


class Alice(Node):
    """
    Version améliorée d'Alice qui utilise un modèle d'apprentissage
    pour prédire les bits des Bobs déconnectés.
    """
    def __init__(self, p_one, buffer_size=20, use_cnn=False):
        super().__init__("Alice", p_one, buffer_size)
        self.bob_history = {}
        self.model = LogisticRegression(random_state=42)
        self.is_model_trained = False
        self.training_data = []
        self.training_labels = []
        
        # Pour l'intégration avec CNN
        self.use_cnn = use_cnn
        self.cnn_model = None
        
    def set_cnn_model(self, model):
        """Configure le modèle CNN pour les prédictions"""
        self.cnn_model = model
        self.use_cnn = True
        
    def receive_message_from_bob(self, bob_id, bit):
        super().receive_message(bit)
        
        if bob_id not in self.bob_history:
            self.bob_history[bob_id] = []
        
        # Applique aussi la limitation de taille de buffer pour l'historique de chaque Bob
        if len(self.bob_history[bob_id]) >= self.buffer_size:
            self.bob_history[bob_id].pop(0)
        
        self.bob_history[bob_id].append(bit)
        
        if not self.use_cnn:
            features = [bob_id, len(self.bob_history[bob_id])]
            self.training_data.append(features)
            self.training_labels.append(bit)
            
            # Limite aussi la taille des données d'entraînement pour garder les plus récentes
            if len(self.training_data) > self.buffer_size * 10:  # Buffer plus grand pour l'apprentissage
                self.training_data.pop(0)
                self.training_labels.pop(0)
    
    def train_model(self):
        """
        Entraîne le modèle de prédiction basé sur les données reçues.
        """
        if self.use_cnn:
            # Le modèle CNN est entraîné séparément
            return True
            
        if len(self.training_data) > 10:
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            try:
                self.model.fit(X, y)
                self.is_model_trained = True
                return True
            except Exception as e:
                print(f"Error training model: {e}")
                return False
        return False
    
    def predict_message(self, bob_id, image_data=None):
        """
        Utilise le modèle entraîné pour prédire le bit d'un Bob déconnecté.
        
        Args:
            bob_id: Identifiant du Bob déconnecté
            image_data: Données d'image partielle pour la prédiction CNN
        """
        if self.use_cnn and self.cnn_model is not None and image_data is not None:
            # Utilise le modèle CNN pour prédire
            with torch.no_grad():
                # L'image_data est déjà sur le bon device (CPU/GPU)
                output = self.cnn_model(image_data)
                # Appliquer sigmoid pour obtenir des probs entre 0 et 1
                probs = torch.sigmoid(output)
                # Récupérer uniquement le pixel correspondant à bob_id
                flat_probs = probs.view(probs.size(0), -1)
                prob = flat_probs[0, bob_id].item()
                
                bit_prediction = 1 if prob >= 0.5 else 0
                return bit_prediction
        else:
            # Utilise la méthode traditionnelle
            if not self.is_model_trained:
                return self.guess_message()
            
            bob_msg_count = len(self.bob_history.get(bob_id, []))
            features = np.array([[bob_id, bob_msg_count]])
            
            try:
                proba = self.model.predict_proba(features)[0][1]
                bit_prediction = 1 if random.random() < proba else 0
                return bit_prediction
            except Exception as e:
                print(f"Error making prediction: {e}")
                return self.guess_message()


# -----------------------------------------------------------------------------
#                     PARTIE 2: DÉFINITION DU CNN AUTOENCODER
# -----------------------------------------------------------------------------

class CNNAutoencoder(nn.Module):
    """
    Un autoencodeur convolutionnel simple:
         - Encoder: 2 blocs Conv -> MaxPool
         - Decoder: 2 blocs ConvTranspose
    """
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Sortie: (B,32,14,14)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # Sortie: (B,64,7,7)
        )
        
        # --- Decoder ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # (B,32,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)    # (B,1,28,28)
            # (FR) Pas de sigmoid ici; on utilisera BCEWithLogitsLoss
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# -----------------------------------------------------------------------------
#                     PARTIE 3: FONCTIONS D'ENTRAÎNEMENT
# -----------------------------------------------------------------------------

def train_cnn_model_with_dynamic_disconnections(model, train_loader, device, epochs=10, disconnect_percentage=15, buffer_size=20, learning_rate=1e-3):
    """
    Entraîne le modèle CNN avec déconnexions dynamiques.
    À chaque lot (batch), un ensemble différent de nœuds est déconnecté.
    
    Args:
        model: Le modèle CNN à entraîner
        train_loader: DataLoader pour les données d'entraînement
        device: Appareil sur lequel exécuter l'entraînement (CPU/GPU)
        epochs: Nombre d'époques d'entraînement
        disconnect_percentage: Pourcentage de nœuds à déconnecter
        buffer_size: Taille du buffer pour les nœuds
        learning_rate: Taux d'apprentissage pour l'optimiseur
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"Debug: Learning rate utilisé: {learning_rate}")
    
    # Calculer la probabilité de déconnexion à partir du pourcentage
    drop_prob = disconnect_percentage / 100.0
    drop_prob_range = (max(0.1, drop_prob - 0.1), min(0.9, drop_prob + 0.1))
    
    print(f"Training CNN model with dynamic disconnections...")
    print(f"Disconnect percentage: {disconnect_percentage}%")
    print(f"Buffer size: {buffer_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0
        
        for batch_images, _ in train_loader:
            # batch_images de forme (B, 1, 28, 28), valeurs {0,1}
            batch_images = batch_images.to(device)
            
            # A) On "aplatie" l'image en (B,784)
            batch_images_flat = batch_images.view(batch_images.size(0), -1)
            
            # Utiliser une probabilité de déconnexion aléatoire à chaque batch
            cur_drop_prob = random.uniform(drop_prob_range[0], drop_prob_range[1])
            
            # B) À chaque batch, différents nœuds (pixels) sont déconnectés
            mask = (torch.rand_like(batch_images_flat) > cur_drop_prob).float()
            dropped_flat = batch_images_flat.clone()
            dropped_flat[mask == 0] = -1

            
            # C) On remet la forme (B,1,28,28) pour le CNN
            dropped_input = dropped_flat.view(batch_images.size(0), 1, 28, 28)
            
            # Forward
            outputs = model(dropped_input)  # (B,1,28,28) [logits]
            
            # Calcul de la loss
            loss = criterion(outputs, batch_images)
            
            # Backprop et update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{epochs}], Train Loss: {avg_loss:.4f}")
    
    print("CNN training complete!")
    # Sauvegarder le modèle avec le pourcentage de déconnexion dans le nom du fichier
    model_filename = f"autoencoder_model_P2P_mnist_disconnect_{disconnect_percentage}pct.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as {model_filename}")
    
    return model


# -----------------------------------------------------------------------------
#                     PARTIE 4: FONCTION PRINCIPALE
# -----------------------------------------------------------------------------

def main():
    """Fonction principale pour l'entraînement du modèle"""
    # Paramètres simples - modifiez directement ces valeurs
    disconnect_percentage = 60  # Pourcentage de Bobs à déconnecter
    buffer_size = 25           # Taille du buffer
    epochs = 15                # Nombre d'époques
    learning_rate = 1e-4     # Taux d'apprentissage
    
    # Afficher les paramètres utilisés
    print("\n=== Paramètres d'entraînement ===")
    print(f"Pourcentage de déconnexion: {disconnect_percentage}%")
    print(f"Taille du buffer: {buffer_size}")
    print(f"Nombre d'époques: {epochs}")
    print(f"Taux d'apprentissage: {learning_rate}")
    print("================================\n")
    
    # Déterminer l'appareil à utiliser (GPU si disponible, sinon CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Définir et charger le dataset MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),  # Binarisation
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 2. Créer les DataLoader
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Créer et configurer le modèle CNN
    cnn_model = CNNAutoencoder().to(device)
    
    # 4. Entraînement du modèle avec déconnexions dynamiques
    cnn_model = train_cnn_model_with_dynamic_disconnections(
        cnn_model, 
        train_loader, 
        device,
        epochs=epochs,
        disconnect_percentage=disconnect_percentage,
        buffer_size=buffer_size,
        learning_rate=learning_rate
    )
    
    print(f"\nEntraînement terminé avec succès!")
    print(f"Le modèle a été sauvegardé avec un taux de déconnexion de {disconnect_percentage}%")


if __name__ == "__main__":
    main()