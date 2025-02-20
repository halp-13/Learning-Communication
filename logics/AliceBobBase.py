import random

class Node:
    """
    Cette classe représente un nœud dans la communication P2P.
    Chaque nœud peut envoyer et recevoir des bits (0 ou 1).
    Il peut également être déconnecté.
    """
    def __init__(self, name, p_one):
        """
        name: Nom du nœud (ex: 'Alice' ou 'Bob')
        p_one: Probabilité qu'un message envoyé par ce nœud soit un bit 1
        """
        self.name = name
        self.p_one = p_one
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
        """
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
        
        # Calcul de la probabilité d'obtenir un bit 1
        prob_one = self.count_ones / self.count_total
        bit_guess = 1 if random.random() < prob_one else 0
        return bit_guess


def simulate_communication(p_alice=0.5, p_bob=0.5, steps=20):
    """
    Lance la simulation d'une communication entre Alice et Bob
    sur un nombre de 'steps'(nombre de messages) défini.
    Bob sera déconnecté à une étape aléatoire entre 1 et 'steps'.
    """
    alice = Node("Alice", p_alice)
    bob   = Node("Bob", p_bob)
    
    # Choisit une étape aléatoire où Bob sera déconnecté
    disconnect_step = random.randint(50, steps)
    
    # Liste pour stocker les bits réels envoyés par Bob
    bob_sent_bits = []
    # Liste pour stocker les bits devinés par Alice après la déconnexion de Bob
    alice_guessed_bits = []
    
    print("\n--- Simulation starts ---")
    print(f"Alice sends '1' with probability {p_alice}.")
    print(f"Bob sends '1' with probability {p_bob}.")
    print(f"Bob will be disconnected at a random step: {disconnect_step}.\n")
    
    for step in range(1, steps + 1):
        print(f"[Step {step}]")
        
        # Vérifie si Bob doit se déconnecter à cette étape
        if step == disconnect_step and not bob.is_disconnected:
            bob.is_disconnected = True
            print("** Bob is now disconnected! **")
        
        # -- Alice --> Bob --
        bit_from_alice = alice.send_message()
        if bit_from_alice is not None:
            # Si Bob n'est pas déconnecté, il reçoit le bit
            if not bob.is_disconnected:
                bob.receive_message(bit_from_alice)
                print(f"Alice sent: {bit_from_alice} --> Bob received.")
            else:
                # Bob est déconnecté, il ne reçoit pas
                print(f"Alice sent: {bit_from_alice}, but Bob is disconnected.")
        else:
            # Dans ce scénario, Alice ne se déconnecte pas,
            # donc ce bloc n'est pas vraiment utilisé.
            print("Alice is disconnected (not handled in this scenario).")
        
        # -- Bob --> Alice --
        bit_from_bob = bob.send_message()
        if bit_from_bob is not None:
            # Bob n'est pas déconnecté, donc Alice reçoit ce bit
            alice.receive_message(bit_from_bob)
            bob_sent_bits.append(bit_from_bob)
            print(f"Bob sent: {bit_from_bob} --> Alice received.")
        else:
            # Bob est déconnecté, donc Alice doit deviner le bit
            guessed_bit = alice.guess_message()
            alice_guessed_bits.append(guessed_bit)
            
            ratio_str = f"{alice.count_ones}/{alice.count_total}" if alice.count_total else "0"
            print(f"Bob is disconnected; Alice guessed using ratio {ratio_str}: {guessed_bit}")
    
    print("\n--- Simulation ends ---")
    print(f"Alice received {alice.count_total} bits from Bob (1s: {alice.count_ones}).")
    print(f"Bob received {bob.count_total} bits from Alice (1s: {bob.count_ones}).")
    
    # Rapport final
    print("\nDetailed report:")
    print(f"Real bits sent by Bob (before disconnection): {bob_sent_bits}")
    print(f"Bits guessed by Alice (after Bob got disconnected): {alice_guessed_bits}\n")
    
    return alice, bob, bob_sent_bits, alice_guessed_bits



if __name__ == "__main__":
    simulate_communication(
        p_alice=0.8,   # Probabilité  Alice
        p_bob=0.5,     # Probabilité  Bob
        steps=100       # Nombre total des messages échangés(chaque message est un bit)
    )

