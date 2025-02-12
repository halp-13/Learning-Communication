import numpy as np

class Bob:
    """Représente Bob, qui envoie des données à Alice."""
    def __init__(self, transmission_length=50, stop_at=30):
        self.transmission_length = transmission_length
        self.stop_at = stop_at
        self.sent_data = np.tile(
            np.array([1]*2 + [0]*(5 - 2), dtype=int),
            self.transmission_length // 5 + 1
        )[:self.transmission_length]
    
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
    def fill_pattern(self, received_length, period_len=6):
        """
        Remplit les bits manquants (depuis 'received_length' jusqu'à la fin de 'transmission_length')
        en répétant un motif de longueur 'period_len'.
        On suppose que les 'period_len' premiers bits reçus constituent le motif de base.
        """
        received_data = self.buffer[:received_length]
        needed = self.transmission_length - received_length
        
        # Hypothèse simple : on considère que le motif de base se trouve dans
        # les 'period_len' premiers bits reçus.
        pattern = received_data[:period_len]
        
        fill = []
        partial_index = received_length % period_len
        
        # Si on se trouve au milieu d'un motif, on complète d'abord la fin de ce motif.
        if partial_index != 0:
            remain_in_cycle = period_len - partial_index
            take = min(remain_in_cycle, needed)
            fill.extend(pattern[partial_index : partial_index + take])
            needed -= take
        
        # On poursuit ensuite en répétant le motif autant que nécessaire.
        while needed > 0:
            if needed >= period_len:
                fill.extend(pattern)
                needed -= period_len
            else:
                fill.extend(pattern[:needed])
                needed = 0
        
        # Finalement, on place le résultat dans 'self.estimated_data'.
        self.estimated_data[:received_length] = received_data
        self.estimated_data[received_length:] = fill
        # Retour
        return self.estimated_data

    
    def estimate_missing_data(self, received_length, period_len=6):
        """
        Remplit d'abord les bits manquants en répétant un motif,
        puis applique un flipping de certains bits afin de s'approcher du ratio de bits à 1
        estimé à partir des données déjà reçues.
        """
        # 1) Remplir les bits restants avec le motif répété
        self.fill_pattern(received_length, period_len=period_len)
        
        # 2) Calculer le ratio estimé à partir des bits reçus
        estimated_ratio = self.estimate_ratio(received_length)
        
        # 3) Calculer le nombre total de 1 nécessaires sur l'ensemble (transmission_length)
        total_ones_needed = int(round(estimated_ratio * self.transmission_length))
        
        # 4) Nombre actuel de 1 dans 'estimated_data'
        current_ones = np.sum(self.estimated_data)
        
        # 5) Calcul de la différence
        difference = total_ones_needed - current_ones
        
        if difference == 0:
        # Le nombre de 1 est déjà correct, on ne fait rien
            return self.estimated_data 
        
        # On ne flippe que dans la partie estimée (non reçue),
        # c’est-à-dire l’intervalle [received_length : transmission_length].
        estimated_part = self.estimated_data[received_length:]
        
        if difference > 0:
            # On doit convertir certains 0 en 1
            zero_indices = np.where(estimated_part == 0)[0]
            if len(zero_indices) == 0:
                return  # Pas de 0 disponible pour le flipping
            num_to_flip = min(difference, len(zero_indices))
            flip_indices = np.random.choice(zero_indices, size=num_to_flip, replace=False)
            estimated_part[flip_indices] = 1
            
        else:  # difference < 0
            # On doit convertir certains 1 en 0
            diff_abs = abs(difference)
            one_indices = np.where(estimated_part == 1)[0]
            if len(one_indices) == 0:
                return  # Pas de 1 disponible pour le flipping
            num_to_flip = min(diff_abs, len(one_indices))
            flip_indices = np.random.choice(one_indices, size=num_to_flip, replace=False)
            estimated_part[flip_indices] = 0
        # Retour
        return self.estimated_data

    
    def send(self):
        """Envoie ses propres données."""
        return self.sent_data

# Initialisation de Bob et Alice
bob = Bob()
alice = Alice()

# Transmission et réception des données
bob_transmitted_data = bob.transmit()
received_length = alice.receive(bob_transmitted_data)

# Estimation des bits manquants (en utilisant le motif + flipping pour conserver le ratio)
final_estimated_data = alice.estimate_missing_data(received_length, period_len=5)

# Estimation du ratio
estimated_ratio = alice.estimate_ratio(received_length)
actual_ratio = np.sum(bob_transmitted_data) / bob.stop_at

# Affichage des résultats
print(f"Ratio réel des bits 1 envoyés par Bob: {actual_ratio:.3f}")
print(f"Ratio estimé par Alice: {estimated_ratio:.3f}")
print(f"Données réelles envoyées par Bob: {bob.sent_data}")
print(f"Données estimées par Alice: {final_estimated_data}")
