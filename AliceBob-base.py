import numpy as np

class Bob:
    def __init__(self, message_length=10):
        """Créer un message binaire aléatoire de longueur spécifiée."""
        self.message_length = message_length
        self.message = np.random.choice([0, 1], size=message_length)
    
    def send_message(self, loss_prob=0.2):
        """
        Envoyer le message avec une probabilité de perte définie (loss_prob).
        Les bits sont envoyés avec une probabilité (1 - loss_prob).
        Les bits perdus sont remplacés par None.
        """
        received_message = []
        for bit in self.message:
            if np.random.rand() > loss_prob:  # Le bit est reçu avec probabilité (1 - loss_prob)
                received_message.append(bit)
            else:
                received_message.append(None)  # Le bit est perdu
        return received_message

class Alice:
    def __init__(self, buffer_size=10):
        """Initialiser Alice avec un buffer de taille fixe."""
        self.buffer_size = buffer_size
        self.buffer = []

    def receive_message(self, received_message):
        """
        Recevoir le message envoyé par Bob et stocker les bits reçus dans le buffer.
        Les bits perdus (None) ne sont pas enregistrés.
        """
        self.buffer = [bit for bit in received_message if bit is not None]

    def predict_lost_bits(self):
        """
        Prédire le ratio des bits 1 perdus en utilisant la formule :
        (Nombre de 1 reçus) / (Taille du buffer)
        """
        if not self.buffer:  # Si aucun bit n'a été reçu
            return 0  # Prédiction de 0
        
        ones_received = sum(self.buffer)  # Compter le nombre de 1 reçus
        prediction = ones_received / self.buffer_size  # Calcul du ratio
        
        return prediction

    def reconstruct_message(self, received_message):
        """
        Reconstruire le message en remplaçant les None par une prédiction.
        Si la prédiction > 0.5, on remplace les None par des 1, sinon par des 0.
        """
        prediction_ratio = self.predict_lost_bits()  # Obtenir la prédiction
        
        # Remplacer les None en fonction de la prédiction
        reconstructed_message = [
            bit if bit is not None else (1 if prediction_ratio > 0.5 else 0)
            for bit in received_message
        ]
        return reconstructed_message

# Exécution de la simulation
bob = Bob(message_length=10)  # Générer un message binaire aléatoire
received_message = bob.send_message(loss_prob=0.3)  # Envoyer le message avec 30% de pertes

alice = Alice(buffer_size=10)  # Initialiser Alice avec un buffer fixe
alice.receive_message(received_message)  # Recevoir les bits envoyés
prediction = alice.predict_lost_bits()  # Faire une prédiction des bits perdus
reconstructed_message = alice.reconstruct_message(received_message)  # Reconstruire le message

# Affichage des résultats
print(f"Message original :      {bob.message}")
print(f"Message reçu :          {received_message}")
print(f"Ratio prédit des 1 perdus : {prediction:.2f}")
print(f"Message reconstruit :   {reconstructed_message}")
