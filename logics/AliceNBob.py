import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
            return 0
        
        # Calcul de la probabilité d'obtenir un bit 1 sur le buffer size 
        # si le buffer est plein, sinon sur le nombre total de bits reçus.
        prob_guess_one = self.count_ones / self.buffer_size if self.count_total >= self.buffer_size else self.count_ones / self.count_total
        bit_guess = 1 if random.random() < prob_guess_one else 0
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
    def __init__(self, p_one, buffer_size=20,model_type = "random_forest"):
        super().__init__("Alice", p_one, buffer_size)
        self.bob_history = {}
        if model_type == "logistic":
            self.model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        elif model_type == "random_forest":           
            self.model = RandomForestClassifier(random_state=42, n_estimators=50)
        else:
            raise ValueError("model_type must be 'logistic' or 'random_forest'")
        self.is_model_trained = False
        self.training_data = []
        self.training_labels = []
        
    def receive_message_from_bob(self, bob_id, bit):
        super().receive_message(bit)
        
        if bob_id not in self.bob_history:
            self.bob_history[bob_id] = []
        
        # Applique aussi la limitation de taille de buffer pour l'historique de chaque Bob
        if len(self.bob_history[bob_id]) >= self.buffer_size:
            self.bob_history[bob_id].pop(0)

        self.bob_history[bob_id].append(bit)
        p_one_bob = sum(self.bob_history[bob_id]) / len(self.bob_history[bob_id])
        
        # Crée les features pour l'entraînement du modèle
        features = [bob_id, p_one_bob, len(self.bob_history[bob_id])]
        
        self.training_data.append(features)
        self.training_labels.append(bit)
        

        # Limite aussi la taille des données d'entraînement pour garder les plus récentes
        if len(self.training_data) > self.buffer_size :
            self.training_data.pop(0)
            self.training_labels.pop(0)
    
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
        # Si le modèle n'est pas encore entraîné ou si le Bob n'a pas d'historique de messages
        if not self.is_model_trained or bob_id not in self.bob_history:
            return self.guess_message()
        # Crée les features pour la prédiction du bit du Bob
        # en prenant en compte les derniers messages reçus par le Bob(limité par buffer_size)
        recent_history = self.bob_history[bob_id][-self.buffer_size:]
        p_one_bob = sum(recent_history) / len(recent_history)
        features = np.array([[bob_id, p_one_bob, len(recent_history)]])
        # Prédiction de la probabilité d'envoyer 1
        try:
            proba = self.model.predict_proba(features)[0][1]
            bit_prediction = 1 if random.random() < proba else 0
            return bit_prediction
        except Exception as e:
            print(f"Error making prediction: {e}")
            return self.guess_message()


def simulate_enhanced_communication(p_alice=0.5, p_bob=0.5, num_bobs=100, disconnect_percentage=20, message_length=100, buffer_size=20, model_type="random_forest"):
    """
    Lance une simulation améliorée avec plusieurs Bobs.
    """
    alice = Alice(p_alice, buffer_size, model_type)
    print(f"Simulation started with the '{model_type}' model.")
    bobs = [Node(f"Bob_{i}", p_bob, buffer_size) for i in range(num_bobs)]
    
    num_disconnected = int(num_bobs * disconnect_percentage / 100)
    disconnected_bobs = random.sample(range(num_bobs), num_disconnected)
    disconnect_steps = {bob_id: random.randint((message_length/4), message_length-1) for bob_id in disconnected_bobs} # Étapes de déconnexion
    
    results = {
        "real_bits": {},
        "predicted_bits": {},
        "would_send_bits": {}
    }
    
    print("\n--- Simulation Started ---")
    print(f"Alice sends '1' with probability {p_alice}.")
    print(f"Each Bob sends '1' with probability {p_bob}.")
    print(f"Total number of Bobs: {num_bobs}")
    print(f"Number of Bobs to be disconnected: {num_disconnected} ({disconnect_percentage}%)")
    print(f"Message length: {message_length} bits\n")
    print(f"Buffer size: {buffer_size}\n")
    
    for bob_id in range(num_bobs):
        results["real_bits"][bob_id] = []
        results["would_send_bits"][bob_id] = []
        if bob_id in disconnected_bobs:
            results["predicted_bits"][bob_id] = []
    
    for step in range(1, message_length + 1):
        if step % 10 == 0 and step > 20:
            alice.train_model()
        
        for bob_id, bob in enumerate(bobs):
            if bob_id in disconnected_bobs and step == disconnect_steps[bob_id] and not bob.is_disconnected:
                bob.is_disconnected = True
            
            bit_from_alice = alice.send_message()
            if bit_from_alice is not None and not bob.is_disconnected:
                bob.receive_message(bit_from_alice)
            
            would_send_bit = 1 if random.random() < p_bob else 0
            
            results["would_send_bits"][bob_id].append(would_send_bit)
            
            bit_from_bob = bob.send_message()
            
            if bit_from_bob is not None:
                alice.receive_message_from_bob(bob_id, bit_from_bob)
                results["real_bits"][bob_id].append(bit_from_bob)
            else:
                predicted_bit = alice.predict_message(bob_id)
                results["predicted_bits"][bob_id].append(predicted_bit)
    
    print("\n--- Simulation Results ---")
    
    total_correct = 0
    total_predictions = 0
    
    for bob_id in disconnected_bobs:
        disconnect_step = disconnect_steps[bob_id]
        predictions = results["predicted_bits"][bob_id]
        would_send = results["would_send_bits"][bob_id][disconnect_step:]
        
        min_length = min(len(predictions), len(would_send))
        predictions = predictions[:min_length]
        would_send = would_send[:min_length]
        
        if min_length > 0:
            correct_predictions = sum(1 for p, w in zip(predictions, would_send) if p == w)
            accuracy = (correct_predictions / min_length) * 100
            ones_in_would_send = sum(would_send) / min_length * 100
            ones_in_predictions = sum(predictions) / min_length * 100
            
            #Cette partie du code permet d’afficher les détails pour chaque Bob déconnecté. 
            #Étant donné le nombre élevé de Bobs (784), 
            #afin d’éviter un trop grand nombre d’affichages lors de l’exécution,  
            #nous avons mis ce bloc en commentaire.  
            #Cependant, si vous souhaitez visualiser ces détails, 
            #il suffit de retirer les triple quotes autour du bloc ci-dessous.  
            '''
            print(f"\nBob_{bob_id}:")
            print(f"  - Disconnected at step: {disconnect_step}")
            print(f"  - Number of predictions: {min_length}")
            print(f"  - Correct predictions: {correct_predictions}")
            print(f"  - Accuracy: {accuracy:.2f}%")
            print(f"  - Probability of being 1 in unsent messages: {ones_in_would_send:.2f}%")
            print(f"  - Probability of being 1 in predicted messages: {ones_in_predictions:.2f}%")
            print(f"  - Difference in the probability of being 1: {abs(ones_in_would_send - ones_in_predictions):.2f}%")
            

            expected_str = ''.join(str(bit) for bit in would_send)
            predicted_str = ''.join(str(bit) for bit in predictions)
            
            print(f"  - Expected bits: {expected_str}")
            print(f"  - Predicted bits: {predicted_str}")
                        
            match_indicators = ''.join(['✓' if e == p else '✗' for e, p in zip(would_send, predictions)])
            print(f"  - Comparison:    {match_indicators}")
            '''
            total_correct += correct_predictions
            total_predictions += min_length
    
    if total_predictions > 0:
        overall_accuracy = (total_correct / total_predictions) * 100
        print(f"\nOverall prediction accuracy: {overall_accuracy:.2f}%")
        # Calculate overall percentage of 1s in all would_send and predicted bits
        all_would_send = []
        all_predictions = []
        for bob_id in disconnected_bobs:
            disconnect_step = disconnect_steps[bob_id]
            predictions = results["predicted_bits"][bob_id]
            would_send = results["would_send_bits"][bob_id][disconnect_step:]
            
            min_length = min(len(predictions), len(would_send))
            all_would_send.extend(would_send[:min_length])
            all_predictions.extend(predictions[:min_length])

        overall_ones_in_would_send = sum(all_would_send) / len(all_would_send) * 100 if all_would_send else 0
        overall_ones_in_predictions = sum(all_predictions) / len(all_predictions) * 100 if all_predictions else 0

        print(f"Overall probability of being 1 in unsent messages: {overall_ones_in_would_send:.2f}%")
        print(f"Overall probability of being 1 in predicted messages: {overall_ones_in_predictions:.2f}%")
        print(f"Overall difference in probabilities: {abs(overall_ones_in_would_send - overall_ones_in_predictions):.2f}%")

        
        naive_correct = int(total_predictions * (1 - p_bob)) if p_bob <= 0.5 else int(total_predictions * p_bob)
        naive_accuracy = (naive_correct / total_predictions) * 100
        print(f"Naive prediction accuracy (based only on p_bob): {naive_accuracy:.2f}%")
        
        if overall_accuracy > naive_accuracy:
            improvement = overall_accuracy - naive_accuracy
            print(f"Improvement from learning model: +{improvement:.2f}%")
        else:
            print("The learning model did not improve predictions compared to naive approach.")
    
    print("\n--- Simulation finished ---")
    
    return alice, bobs, results


if __name__ == "__main__":
    # pour choisir le modèle à utiliser, décommentez la ligne correspondante
    model_type = "random_forest"
    #model_type = "logistic"
    simulate_enhanced_communication(
        p_alice=0.7, # probabilité d'envoyer 1 pour Alice
        p_bob=0.5, # probabilité d'envoyer 1 pour Bob
        num_bobs=784,
        disconnect_percentage=15, # 15% des Bobs seront déconnectés
        message_length=1000, # Nombre total de messages échangés (chaque message est un bit)
        buffer_size=40, # Taille du buffer pour chaque nœud
        model_type=model_type
    )