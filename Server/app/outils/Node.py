import random
import numpy as np
from sklearn.linear_model import LogisticRegression


class Node:
    """
    Cette classe représente un nœud dans la communication P2P.
    Chaque nœud peut envoyer et recevoir des bits (0 ou 1).
    Il peut également être déconnecté.
    """
    def __init__(self, name, p_one, buffer_size=20):
        """
        name: Nom du nœud (ex: 'Alice' ou 'Bob')
        p_one: Probabilité qu'un message envoyé par ce nœud soit un bit 1
        buffer_size: Taille du buffer pour stocker les bits reçus
        """
        self.name = name
        self.p_one = p_one
        self.buffer_size = buffer_size # Taille du buffer
        self.is_disconnected = False  # Statut de déconnexion
        self.received_bits = []       # Bits reçus de l'autre nœud
        self.count_ones = 0          # Nombre de bits 1 reçus
        self.count_total = 0         # Nombre total de bits reçus
    
    def send_message(self):
        """
        Envoie un bit (0 ou 1) si le nœud n'est pas déconnecté.
        Retourne None si le nœud est déconnecté.
        """
        if self.is_disconnected:
            return None
        # Choisit un bit = 1 avec probabilité p_one
        bit = 1 if random.random() < self.p_one else 0
        return bit
    
    def receive_message(self, bit):
        """
        Reçoit un bit et met à jour les statistiques.
        Si le buffer est plein, retire le bit le plus ancien avant d'ajouter le nouveau bit.
        """
        # Si le buffer est plein, on retire le bit le plus ancien
        if len(self.received_bits) >= self.buffer_size:
            oldest_bit = self.received_bits.pop(0) # Retire le bit le plus ancien
            if oldest_bit == 1:
                self.count_ones -= 1
            self.count_total -= 1
        # Met à jour les statistiques
        self.received_bits.append(bit)
        self.count_total += 1
        if bit == 1:
            self.count_ones += 1
    
    def guess_message(self):
        """
        Devine un bit lorsque l'autre nœud est déconnecté et ne renvoie rien.
        La probabilité d'être 1 = (count_ones / count_total).
        Retourne 0 par défaut si aucun bit n'a été reçu (count_total=0).
        """
        if self.count_total == 0:
            return 0  # Pas de donnée statistique, on renvoie 0 par défaut.
        
        # Calcul de la probabilité d'obtenir un bit 1 sur le buffer size 
        # si le buffer est plein, sinon sur le nombre total de bits reçus.
        prob_one = self.count_ones / self.buffer_size if self.count_total >= self.buffer_size else self.count_ones / self.count_total
        bit_guess = 1 if random.random() < prob_one else 0
        return bit_guess
    

class Alice(Node):
    """
    Version améliorée d'Alice qui utilise un modèle d'apprentissage
    pour prédire les bits des Bobs déconnectés.
    """
    def __init__(self, p_one):
        super().__init__("Alice", p_one)
        self.bob_history = {}
        self.model = LogisticRegression(random_state=42)
        self.is_model_trained = False
        self.training_data = []
        self.training_labels = []
        
    def receive_message_from_bob(self, bob_id, bit):
        super().receive_message(bit)
        
        if bob_id not in self.bob_history:
            self.bob_history[bob_id] = []
        
        self.bob_history[bob_id].append(bit)
        
        features = [bob_id, len(self.bob_history[bob_id])]
        self.training_data.append(features)
        self.training_labels.append(bit)
    
    def train_model(self):
        """
        Entraîne le modèle de prédiction basé sur les données reçues.
        """
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
    
    def predict_message(self, bob_id):
        """
        Utilise le modèle entraîné pour prédire le bit d'un Bob déconnecté.
        """
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