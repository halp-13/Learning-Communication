import numpy as np

class Bob:
    def __init__(self, original_message):
        self.message = original_message # Utiliser le message original
    """def __init__(self, message_length=10):
        #Créer un message binaire aléatoire de longueur spécifiée pour chaque Bob.
        self.message_length = message_length
        self.message = np.random.choice([0, 1], size=message_length)"""
    
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
    def __init__(self, buffer_size=10, num_bobs=3):
        """Initialiser Alice avec un buffer de taille fixe et plusieurs Bob."""
        self.buffer_size = buffer_size
        self.num_bobs = num_bobs
        self.buffers = [[] for _ in range(num_bobs)]  # Buffer séparé pour chaque Bob

    def receive_messages(self, received_messages):
        """
        Recevoir les messages envoyés par plusieurs Bob.
        Stocker les bits reçus pour chaque Bob dans un buffer dédié.
        """
        for i, received_message in enumerate(received_messages):
            self.buffers[i] = [bit for bit in received_message if bit is not None]

    def reconstruct_message(self, received_messages):
        """
        Reconstruire le message en combinant les données reçues de plusieurs Bob.
        Si un bit est perdu chez un Bob, on regarde les autres Bob.
        Si le bit est perdu partout, on le prédit comme avant.
        """
        message_length = len(received_messages[0])  # Longueur des messages
        reconstructed_message = []

        for i in range(message_length):
            available_bits = [received_messages[j][i] for j in range(self.num_bobs) if received_messages[j][i] is not None]

            if available_bits:  # Si au moins un Bob a envoyé ce bit
                reconstructed_message.append(available_bits[0])  # Prendre le premier disponible
            else:
                # Prédire le bit perdu en utilisant la méthode précédente
                prediction_ratio = self.predict_lost_bits()
                predicted_bit = 1 if prediction_ratio > 0.5 else 0
                reconstructed_message.append(predicted_bit)

        return reconstructed_message

    def predict_lost_bits(self):
        """
        Prédire le ratio des bits 1 perdus en utilisant la formule :
        (Nombre de 1 reçus) / (Taille du buffer total)
        """
        total_received_ones = sum(sum(buffer) for buffer in self.buffers if buffer)
        total_received_bits = sum(len(buffer) for buffer in self.buffers)

        if total_received_bits == 0:
            return 0  # Si aucun bit n'est reçu

        return total_received_ones / total_received_bits

# Nombre de Bob
num_bobs = 3

# Création des Bob et le message
original_message = np.random.choice([0, 1], size=20)  # Message binaire aléatoire
bobs = [Bob(original_message) for _ in range(num_bobs)] # Création de plusieurs Bob
#bobs = [Bob(message_length=10) for _ in range(num_bobs)]  # Création de plusieurs Bob avec des messages diferentes

# Simulation de l'envoi des messages
received_messages = [bob.send_message(loss_prob=0.5) for bob in bobs]

# Initialisation d'Alice
alice = Alice(buffer_size=20, num_bobs=num_bobs)

# Alice reçoit les messages
alice.receive_messages(received_messages)

# Alice reconstruit le message
reconstructed_message = alice.reconstruct_message(received_messages)

# Affichage des résultats
for i, bob in enumerate(bobs):
    print(f"Message de Bob {i+1} :      {bob.message}")

print("\nMessages reçus (avec pertes) :")
for i, received_message in enumerate(received_messages):
    print(f"Bob {i+1} : {received_message}")

print(f"\nMessage reconstruit par Alice : {reconstructed_message}")
