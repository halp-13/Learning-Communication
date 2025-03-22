import random
import matplotlib.pyplot as plt

class Node:
    Nodes = {}


    def __init__(self, node_id, success_probability, buffer_size=20, learning_rate=0.001):
        """
        Initialise un nœud.
        
        :param node_id: Identifiant unique du nœud.
        :param success_probability: Probabilité de succès d'envoi d'un message (entre 0 et 1).
        :param buffer_size: Taille maximale du buffer pour stocker les messages reçus.
        :param learning_rate: Taux d'apprentissage pour la descente de gradient.
        """
        self.node_id = node_id
        self.success_probability = success_probability
        self.buffer_size = buffer_size
        self.buffer = []  # Buffer pour stocker les messages reçus
        self.connected_nodes = []  # Liste des nœuds connectés
        self.p_values = {}  # Dictionnaire pour stocker les estimations de p-value des autres nœuds
        self.learning_rate = learning_rate  # Taux d'apprentissage pour la descente de gradient
        self.loss_history = []  # Historique des erreurs (loss)
        Node.Nodes[node_id] = self

    def connect(self, other_node):
        """
        Connecte ce nœud à un autre nœud.
        
        :param other_node: Un autre objet Node auquel ce nœud se connecte.
        """
        if other_node not in self.connected_nodes:
            self.connected_nodes.append(other_node)
            self.p_values[other_node.node_id] = 0.5  # Initialiser la p-value à 0.5 (aucune information)

    def send_message(self):
        """
        Simule l'envoi d'un message avec une probabilité de succès donnée.
        
        :return: 1 si le message est envoyé avec succès, 0 sinon.
        """
        if random.random() < self.success_probability:
            return 1  # Succès
        else:
            return 0  # Échec

    def receive_message(self, message, sender_id):
        """
        Reçoit un message et l'ajoute au buffer si celui-ci n'est pas plein.
        
        :param message: Le message reçu.
        :param sender_id: L'identifiant du nœud envoyeur.
        """
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((sender_id, message))
        else:
            # Retirer le message le plus ancien
            self.buffer.pop(0)
            self.buffer.append((sender_id, message))

    def compute_error(self, sender_id, observed_value):
        """
        Calcule l'erreur entre la p-value estimée et la valeur observée.
        
        :param sender_id: L'identifiant du nœud envoyeur.
        :param observed_value: La valeur observée (0 ou 1).
        :return: L'erreur.
        """
        predicted_p = self.p_values.get(sender_id, 0.5)
        error = observed_value - predicted_p
        return error

    def gradient_descent_update(self, sender_id, error):
        """
        Met à jour la p-value estimée en utilisant la descente de gradient.
        
        :param sender_id: L'identifiant du nœud envoyeur.
        :param error: L'erreur calculée.
        """
        if sender_id in self.p_values:
            self.p_values[sender_id] += self.learning_rate * error
            # S'assurer que la p-value reste dans l'intervalle [0, 1]
            self.p_values[sender_id] = max(0, min(1, self.p_values[sender_id]))

    def process_turn(self):
        """
        Traite un tour complet pour ce nœud :
        1. Envoie un message à tous les nœuds connectés.
        2. Reçoit les messages des nœuds connectés.
        3. Estime les p-values et met à jour avec la descente de gradient.
        4. Calcule et affiche la loss.
        """
        # Envoyer un message à tous les nœuds connectés
        send_result = self.send_message()
        print(f"Nœud {self.node_id} a envoyé un message -> Résultat: {send_result}")

        # Recevoir les messages des nœuds connectés
        for other_node in self.connected_nodes:
            other_node.receive_message(send_result, self.node_id)

        # Estimer les p-values et mettre à jour avec la descente de gradient
        total_loss = 0
        for other_node in self.connected_nodes:
            messages_from_node = [msg for sender_id, msg in self.buffer if sender_id == other_node.node_id]
            if messages_from_node:
                observed_value = messages_from_node[-1]  # Dernier message reçu
                error = self.compute_error(other_node.node_id, observed_value)
                self.gradient_descent_update(other_node.node_id, error)
                total_loss += error ** 2  # Loss = erreur au carré (MSE)

        # Stocker la loss moyenne pour ce tour
        if self.connected_nodes:
            average_loss = total_loss / len(self.connected_nodes)
            self.loss_history.append(average_loss)
            print(f"Nœud {self.node_id} - Loss moyenne: {average_loss:.4f}")

        # Afficher l'état du buffer et les p-values estimées
        print(f"Nœud {self.node_id} - Buffer: {self.buffer}")
        print(f"Nœud {self.node_id} - Estimations des p-values: {self.p_values}")

# Fonction pour afficher les loss avec matplotlib
def plot_loss_history(nodes):
    """
    Affiche l'historique des loss pour chaque nœud.
    
    :param nodes: Liste des nœuds.
    """
    plt.figure(figsize=(10, 6))
    for node in nodes:
        plt.plot(node.loss_history, label=f"Nœud {node.node_id}")
    plt.xlabel("Tours")
    plt.ylabel("Loss moyenne")
    plt.title("Évolution de la loss moyenne par nœud")
    plt.legend()
    plt.grid(True)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Créer des nœuds
    node1 = Node(node_id=1, success_probability=1/3, learning_rate=0.001)
    node2 = Node(node_id=2, success_probability=1/2, learning_rate=0.01)
    node3 = Node(node_id=3, success_probability=2/3, learning_rate=0.01)

    # Connecter les nœuds
    node1.connect(node2)
    node1.connect(node3)
    node2.connect(node1)
    node2.connect(node3)
    node3.connect(node1)
    node3.connect(node2)

    # Simuler plusieurs tours
    num_turns = 1000
    for turn in range(num_turns):
        print(f"\n--- Tour {turn + 1} ---")
        node1.process_turn()
        node2.process_turn()
        node3.process_turn()

    # Afficher les loss avec matplotlib
    plot_loss_history([node1, node2, node3])

    # Comparer les p-values estimées avec les vraies p-values
    print("\n--- Comparaison des p-values ---")
    true_p_values = {
        1: 1/3,
        2: 1/2,
        3: 2/3
    }
    for node in [node1, node2, node3]:
        print(f"\nNœud {node.node_id} - Comparaison des p-values:")
        for other_node_id, estimated_p in node.p_values.items():
            true_p = true_p_values[other_node_id]
            print(f"  Nœud {other_node_id} - Estimé: {estimated_p:.4f}, Vrai: {true_p:.4f}, Erreur: {abs(estimated_p - true_p):.4f}")