import numpy as np

class Bob:
    """Représente Bob, qui envoie des données à Alice."""
    def __init__(self, transmission_length=50, stop_at=30):
        self.transmission_length = transmission_length
        self.stop_at = stop_at
        self.sent_data = np.random.randint(0, 2, size=transmission_length)
    
    def transmit(self):
        """Transmet les données jusqu'à stop_at bits."""
        return self.sent_data[:self.stop_at]

class Alice:
    """Représente Alice, qui reçoit des données de Bob, envoie ses propres données et estime les bits manquants."""
    def __init__(self, buffer_size=100, transmission_length=50):
        self.buffer_size = buffer_size
        self.transmission_length = transmission_length
        self.buffer = np.zeros(transmission_length, dtype=int)
        self.sent_data = np.random.randint(0, 2, size=transmission_length)
        self.estimated_data = np.zeros(transmission_length, dtype=int)
    
    def receive(self, data):
        """Reçoit les données envoyées par Bob et les stocke dans son tampon."""
        received_length = min(self.transmission_length, len(data))
        self.buffer[:received_length] = data[:received_length]
        return received_length
    
    def estimate_ratio(self, received_length):
        """Estime le ratio des bits 1 dans les données reçues."""
        ones_received = np.sum(self.buffer[:received_length])
        return ones_received / received_length if received_length > 0 else 0
    
    def estimate_missing_data(self, received_length):
        """Estime les bits manquants après l'arrêt de Bob en utilisant le ratio estimé."""
        estimated_ratio = self.estimate_ratio(received_length)
        estimated_bits = np.random.choice([0, 1], size=self.transmission_length-received_length, p=[1-estimated_ratio, estimated_ratio])
        self.estimated_data[:received_length] = self.buffer[:received_length]
        self.estimated_data[received_length:] = estimated_bits
    
    def send(self):
        """Envoie ses propres données."""
        return self.sent_data

# Initialisation de Bob et Alice
bob = Bob()
alice = Alice()

# Transmission et réception des données
bob_transmitted_data = bob.transmit()
received_length = alice.receive(bob_transmitted_data)

# Estimation des bits manquants
alice.estimate_missing_data(received_length)

# Estimation du ratio
estimated_ratio = alice.estimate_ratio(received_length)
actual_ratio = np.sum(bob_transmitted_data) / bob.stop_at

# Affichage des résultats
print(f"Ratio réel des bits 1 envoyés par Bob: {actual_ratio:.3f}")
print(f"Ratio estimé par Alice: {estimated_ratio:.3f}")
print(f"Données réelles envoyées par Bob: {bob.sent_data}")
print(f"Données estimées par Alice: {alice.estimated_data}")
