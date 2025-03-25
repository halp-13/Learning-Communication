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


class Node_Mis:
    Nodes = {}  # Dictionnaire pour stocker tous les nœuds
    p_values = {}

    def __init__(self, node_id, p_one_value, buffer_size=20, learning_rate=0.001):
        """
        Initialise un nœud.
        
        :param node_id: Identifiant unique du nœud.
        :param p_one_value: Probabilité de succès d'envoi d'un message (entre 0 et 1).
        :param buffer_size: Taille maximale du buffer pour stocker les messages reçus.
        :param learning_rate: Taux d'apprentissage pour la descente de gradient.
        """
        self.node_id = node_id
        self.p_one_value = p_one_value
        self.buffer_size = buffer_size
        self.buffer = {}
        self.guess_buffer = {}
        self.connected_nodes = []  # Liste des nœuds connectés
        self.p_values = {}  # Dictionnaire pour stocker les estimations de p-value des autres nœuds
        self.learning_rate = learning_rate  # Taux d'apprentissage pour la descente de gradient
        self.loss_history = {}  # Historique des erreurs locales
        Node_Mis.Nodes[node_id] = self
        Node_Mis.p_values[node_id] = 0.5  # Initialiser la p-value à 0.5 (aucune information)
        self.sents_message=[]

    @staticmethod
    def reset():
            """
            Réinitialise les nœuds.
            """
            Node_Mis.Nodes = {}
            Node_Mis.p_values = {}

    def connect(self, other_node):
        """
        Connecte ce nœud à un autre nœud.
        
        :param other_node: Un autre objet Node auquel ce nœud se connecte.
        """
        if other_node not in self.connected_nodes:
            self.connected_nodes.append(other_node)
            self.p_values[other_node.node_id] = 0.5  # Initialiser la p-value à 0.5 (aucune information)

    def send_message_p_value(self,p_one_value):
        """
        Simule l'envoi d'un message avec une probabilité de succès donnée.
        
        :return: 1 si le message est envoyé avec succès, 0 sinon.
        """
        if random.random() < p_one_value:
            return 1  # Succès
        else:
            return 0  # Échec
        
    def send_message_p_value_buffer(self,p_one_value,buffer):

        if not buffer:
            bit = self.send_message_p_value(p_one_value)
            buffer.append(bit)
            return bit

        actual_pvalue_0= sum(buffer)/(len(buffer)+1)
        actual_pvalue_1= (sum(buffer)+1)/(len(buffer)+1)

        distance_0= abs(p_one_value-actual_pvalue_0)
        distance_1= abs(p_one_value-actual_pvalue_1)

        bit = 0
        if distance_0 < distance_1:
            bit= 0
        else:
            bit = 1
            self.sents_message.append(bit)
        return bit
    
    def send_message(self):

        return self.send_message_p_value(self.p_one_value)
        # return self.send_message_p_value_buffer(self.p_one_value,self.sents_message[-self.buffer_size:])
        
    def guess_node_bit(self,node_id):
        if node_id not in self.guess_buffer:
            self.guess_buffer[node_id] = self.buffer.get(node_id, [])
        

            
        p_value = self.p_values.get(node_id, 0.5)
        buffer = self.guess_buffer[node_id]
        
        bit = self.guess_bit(p_value,buffer)
        if len(buffer) >= self.buffer_size:
            self.guess_buffer[node_id].pop(0)
        self.guess_buffer[node_id].append(bit)
   
        return bit
        
    def guess_bit(self,p_one_value,buffer):
        """
        Devine le bit d'un nœud en fonction des p-values estimées des nœuds connectés.
        """
        if not buffer:
            return self.send_message_p_value(p_one_value)
        actual_pvalue_0= sum(buffer)/(len(buffer)+1)
        actual_pvalue_1= (sum(buffer)+1)/(len(buffer)+1)

        distance_0= abs(p_one_value-actual_pvalue_0)
        distance_1= abs(p_one_value-actual_pvalue_1)

        if distance_0 < distance_1:
            return 0
        else:
            return 1

    def receive_message(self, message, sender_id):
        """
        Reçoit un message et l'ajoute au buffer si celui-ci n'est pas plein.
        
        :param message: Le message reçu.
        :param sender_id: L'identifiant du nœud envoyeur.
        """

        messages = self.buffer.get(sender_id, [])
        if len(messages) >= self.buffer_size:
            messages.pop(0)

        messages.append(message)
        self.buffer[sender_id] = messages
        
    def compute_error(self, sender_id):
        """
        Calcule l'erreur entre la p-value estimée et la valeur observée.
        
        :param sender_id: L'identifiant du nœud envoyeur.
        :param observed_value: La valeur observée (0 ou 1).
        :return: L'erreur.
        """
        if sender_id not in self.buffer or len(self.buffer[sender_id]) == 0:
            return 0
        
        true_p = sum(self.buffer[sender_id]) / len(self.buffer[sender_id])

        predicted_p = self.p_values.get(sender_id, 0.5)

        error = true_p - predicted_p

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

        else:
            print(f"Node {self.node_id} has no p-value for node {sender_id}")

    def guess_other_node(self):
        """
        Devine la valeur d'un autre nœud en fonction des p-values estimées des nœuds connectés.
        """
        guess={}
        for other_node in self.connected_nodes:
            guess[other_node.node_id]= self.guess_bit(other_node.p_one_value,self.buffer.get(other_node.node_id, []))
        return guess
    
    @staticmethod
    def connect_nodes(list_nodes):
        """
        Connecte les nœuds entre eux.
        """
        for node1, node2 in list_nodes:
            Node_Mis.Nodes[node1].connect(Node_Mis.Nodes[node2])
            Node_Mis.Nodes[node2].connect(Node_Mis.Nodes[node1])

    @staticmethod
    def turn():
        """
        Traite un tour complet pour tous les nœuds.
        """
    

        for node in Node_Mis.Nodes.values():
            node.process_turn()

    @staticmethod
    def vote(node_id):
        """
        Tout les nodes connecté à un node guess le bit de ce node et la majorité l'emporte
        """
        node = Node_Mis.Nodes[node_id]
        votes = []
        for n in  node.connected_nodes :
            bit = n.guess_node_bit(node_id)
            votes.append(bit)

        return max(set(votes), key=votes.count)

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
        
        # print(f"Nœud {self.node_id} a envoyé un message -> Résultat: {send_result}")

        # Recevoir les messages des nœuds connectés
        for other_node in self.connected_nodes:
            other_node.receive_message(send_result, self.node_id)

        # Estimer les p-values et mettre à jour avec la descente de gradient
        self.learn()

    def learn(self):
        
        total_loss = 0
        for other_node in self.connected_nodes:
            error = self.compute_error(other_node.node_id)
            self.gradient_descent_update(other_node.node_id, error)
            total_loss += error ** 2  # Loss = erreur au carré (MSE)
            if other_node.node_id not in self.loss_history:
                self.loss_history[other_node.node_id] = []
            self.loss_history[other_node.node_id].append(error ** 2)
                
    @staticmethod
    def get_node(node_id):
        return Node_Mis.Nodes[node_id]

 
    @staticmethod
    def plot_loss_historys():
        """
        Affiche l'historique des erreurs globales avec matplotlib.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(Node_Mis.loss_history, label="Loss moyenne globale")
        plt.xlabel("Tours")
        plt.ylabel("Loss")
        plt.title("Évolution de la loss moyenne globale")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_loss_history(self):
        """
        Affiche l'historique des erreurs locales avec matplotlib.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label=f"Loss moyenne - Nœud {self.node_id}")
        plt.xlabel("Tours")
        plt.ylabel("Loss")
        plt.title(f"Évolution de la loss moyenne - Nœud {self.node_id}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_loss_history_all(self):
        """
        Affiche l'historique des erreurs locales avec matplotlib.
        """
        plt.figure(figsize=(10, 6))
        for node_id, loss_history in self.loss_history.items():
            plt.plot(loss_history, label=f"Loss moyenne - Nœud {node_id}")
        plt.xlabel("Tours")
        plt.ylabel("Loss")
        plt.title(f"Évolution de la loss moyenne - Nœuds")
        plt.legend()
        plt.grid(True)
        plt.show()