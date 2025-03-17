from django.shortcuts import render
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
import base64

from django.shortcuts import render
from django.http import StreamingHttpResponse
import time
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import random
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier



from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
import os
import pickle

from .ml_model.misGmnist import CNNAutoencoder as misGmnist ,create_geometric_graph_structure, apply_node_failures, create_binary_image, compute_mis


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatternPredictor:
    def __init__(self, buffer_size=20):
        self.buffer_size = buffer_size
    
    

    
    def predict_next_bit(self, history):
        if len(history) < self.buffer_size:
            return random.randint(0, 1)
            
        history_str = ''.join(map(str, history))
        pattern_counts = {}
        
        for length in range(1, self.buffer_size // 2 + 1):
            sub_pattern = history_str[-length:]
            count = history_str[:-length].count(sub_pattern)
            if count > 0:
                pattern_counts[sub_pattern] = count
        
        if pattern_counts:
            best_pattern = max(pattern_counts, key=pattern_counts.get)
            next_index = history_str.rfind(best_pattern) + len(best_pattern)
            if next_index < len(history):
                return self.history[next_index]
        
        return random.randint(0, 1)

class Node:
    def __init__(self, name, p_one=0.5, send_mode="random", guess_mode="random", buffer_size=20):
        self.name = name
        self.p_one = p_one
        self.is_disconnected = False
        self.received_bits = []  # Liste plate pour stocker les bits
        self.count_ones = 0
        self.count_total = 0
        self.send_mode = send_mode
        self.guess_mode = guess_mode
        self._send_counter = 0
        self.buffer_size = buffer_size
        self.pattern_predictor = PatternPredictor(buffer_size)

    def send_message(self):
        if self.is_disconnected:
            return None

        if self.send_mode == "random":
            return self._send_random()
        elif self.send_mode == "alternate":
            return self._send_alternate()
        elif self.send_mode == "pattern":
            return self._send_pattern()
        else:
            raise ValueError(f"Mode d'envoi inconnu : {self.send_mode}")

    def _send_random(self):
        return 1 if random.random() < self.p_one else 0

    def _send_alternate(self):
        bit = self._send_counter % 2
        self._send_counter += 1
        return bit

    def _send_pattern(self):
        pattern = [1, 0, 1, 1, 0]  # Motif personnalisé
        bit = pattern[self._send_counter % len(pattern)]
        self._send_counter += 1
        return bit

    def receive_message(self, bit):
        self.received_bits.append(bit)  # Ajouter un bit à la liste plate
        self.count_total += 1
        if bit == 1:
            self.count_ones += 1

    # def guess_message(self):
    #     if self.guess_mode == "random":
    #         return self._guess_random()
    #     elif self.guess_mode == "probability":
    #         return self._guess_probability()
    #     elif self.guess_mode == "perceptron":
    #         return self._guess_perceptron()
    #     else:
    #         raise ValueError(f"Mode de prédiction inconnu : {self.guess_mode}")

    def _guess_random(self):
        return random.randint(0, 1)

    def _guess_probability(self):
        if self.count_total == 0:
            return 0
        prob_one = self.count_ones / self.count_total
        return 1 if random.random() < prob_one else 0



    def guess_message(self):
        if self.guess_mode == "random":
            return random.randint(0, 1)
        elif self.guess_mode == "probability":
            if self.count_total == 0:
                return 0
            prob_one = self.count_ones / self.count_total
            return 1 if random.random() < prob_one else 0
        elif self.guess_mode == "pattern":
            return self.pattern_predictor.predict_next_bit(self.received_bits)
        else:
            raise ValueError(f"Mode de prédiction inconnu : {self.guess_mode}")


    def to_dict(self):
        return {
            "name": self.name,
            "p_one": self.p_one,
            "is_disconnected": self.is_disconnected,
            "received_bits": self.received_bits,
            "count_ones": self.count_ones,
            "count_total": self.count_total,
            "send_mode": self.send_mode,
            "guess_mode": self.guess_mode
        }
    
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2))

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def index(request):
    return render(request, 'index.html')







@csrf_exempt
def AliceBob(request):
    "Just 2 Nodes Communication (alice and bob) afeter bob disconnected alice will guess the message"
    # mode = 'randomn', ,message=''
    data = request.GET
    p_alice = float(data.get('p_alice', 0.5))
    p_bob = float(data.get('p_bob', 0.5))
    message_length = int(data.get('message_length', 100))
    disconnect_percentage = int(data.get('disconnect_percentage', 20))
    mode = data.get('mode', 'random')
    message = data.get('message', '')

    # return JsonResponse({"message": "Hello, world!"}) 

    num_disconnected = int(message_length * disconnect_percentage / 100)

    def Alice_Bob(p_alice=0.5, p_bob=0.5, message_length=100):
      
        alice = Node("Alice", p_alice, send_mode="alternate", guess_mode="pattern")
        bob = Node("Bob", p_bob , send_mode="pattern", guess_mode="probability")

        results = {
            "real_bits": [],
            "predicted_bits": []
        }



        for step in range(1, message_length + 1):
            disconnected = False if step < num_disconnected else True
            bit_from_alice = alice.send_message()
            bit_from_bob = bob.send_message()
            guess = "."
            if not disconnected :
                alice.receive_message(bit_from_bob)
                bob.receive_message(bit_from_alice)
            else:
                guess = alice.guess_message()

                

     

            results["real_bits"].append(bit_from_bob)
            
            results["predicted_bits"].append(guess)
            

            message = json.dumps({"step": step,"alice":bit_from_alice,"bob":bit_from_bob ,"guess":guess, "disconnected":disconnected }) + "\n"
            yield message
            time.sleep(1)



        correct_predictions = sum(1 for real, predicted in zip(results["real_bits"], results["predicted_bits"]) if real == predicted)
        accuracy = (correct_predictions / message_length) * 100

        yield json.dumps({"correct_predictions":correct_predictions, "accuracy":accuracy}) + "\n"
 


    return StreamingHttpResponse(Alice_Bob(p_alice,p_bob,message_length), content_type="application/json")

@csrf_exempt
def AliceNBob(request):

    # get the data from the request
    data = request.GET
    p_alice = float(data.get('p_alice', 0.5))
    p_bob = float(data.get('p_bob', 0.5))
    num_bobs = int(data.get('nb_bob', 100))
    disconnect_percentage = int(data.get('disconnect_percentage', 20))
    message_length = int(data.get('message_length', 100))


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

    def simulate_enhanced_communication(p_alice=0.5, p_bob=0.5, num_bobs=100, disconnect_percentage=20, message_length=100):
        """
        Lance une simulation améliorée avec plusieurs Bobs.
        """
        alice = Alice(p_alice)
        bobs = [Node(f"Bob_{i}", p_bob) for i in range(1,num_bobs+1)]
        
        num_disconnected = int(num_bobs * disconnect_percentage / 100)
        disconnected_bobs = random.sample(range(1,num_bobs+1), num_disconnected)
        disconnect_steps = {bob_id: random.randint(1, message_length-1) for bob_id in disconnected_bobs} # Étapes de déconnexion
        
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
        
        for bob_id in range(1,num_bobs+1):
            results["real_bits"][bob_id] = []
            results["would_send_bits"][bob_id] = []
            if bob_id in disconnected_bobs:
                results["predicted_bits"][bob_id] = []
        
        for step in range(1, message_length + 1):
            bit_from_alices = {}
            bit_from_bobs = {}
            would_send_bits = {}
            disconnecteds = []

            if step % 10 == 0 and step > 20:
                alice.train_model()
            

            
            for bob_id, bob in enumerate(bobs):
                bob_id += 1
                if bob_id in disconnected_bobs and step == disconnect_steps[bob_id] and not bob.is_disconnected:
                    bob.is_disconnected = True
                    print(f"** Bob_{bob_id} is now disconnected (step {step})! **")
                    disconnecteds.append(bob_id)

                bit_from_alice = alice.send_message()
                if bit_from_alice is not None and not bob.is_disconnected:
                    bob.receive_message(bit_from_alice)
                    bit_from_alices[bob_id] = bit_from_alice

                
                would_send_bit = 1 if random.random() < p_bob else 0
                results["would_send_bits"][bob_id].append(would_send_bit)
                would_send_bits[bob_id] = would_send_bit


                
                bit_from_bob = bob.send_message()
                bit_from_bobs[bob_id] = bit_from_bob
                

                if bit_from_bob is not None:
                    alice.receive_message_from_bob(bob_id, bit_from_bob)
                    results["real_bits"][bob_id].append(bit_from_bob)
                else:
                    predicted_bit = alice.predict_message(bob_id)
                    results["predicted_bits"][bob_id].append(predicted_bit)
            
            if step % 25 == 0 or step == message_length:
                print(f"[Progress]: {step}/{message_length} steps completed.")

            message = json.dumps({"step": step , "alice":bit_from_alices, "bob":bit_from_bobs, "would_send":would_send_bits,"disconnecteds":disconnecteds}) + "\n"
            yield message
            time.sleep(1)
        
        print("\n--- Simulation Finished ---")
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
                
                print(f"\nBob_{bob_id}:")
                print(f"  - Disconnected at step: {disconnect_step}")
                print(f"  - Number of predictions: {min_length}")
                print(f"  - Correct predictions: {correct_predictions}")
                print(f"  - Accuracy: {accuracy:.2f}%")
                

                expected_str = ''.join(str(bit) for bit in would_send)
                predicted_str = ''.join(str(bit) for bit in predictions)
                
                print(f"  - Expected bits: {expected_str}")
                print(f"  - Predicted bits: {predicted_str}")
                            
                match_indicators = ''.join(['✓' if e == p else '✗' for e, p in zip(would_send, predictions)])
                print(f"  - Comparison:    {match_indicators}")
                
                total_correct += correct_predictions
                total_predictions += min_length
        
        if total_predictions > 0:
            overall_accuracy = (total_correct / total_predictions) * 100
            print(f"\nOverall prediction accuracy: {overall_accuracy:.2f}%")
            
            naive_correct = int(total_predictions * (1 - p_bob)) if p_bob <= 0.5 else int(total_predictions * p_bob)
            naive_accuracy = (naive_correct / total_predictions) * 100
            print(f"Naive prediction accuracy (based only on p_bob): {naive_accuracy:.2f}%")
            
            if overall_accuracy > naive_accuracy:
                improvement = overall_accuracy - naive_accuracy
                print(f"Improvement from learning model: +{improvement:.2f}%")
            else:
                print("The learning model did not improve predictions compared to naive approach.")
        
        print("\n--- Simulation finished ---")
        
        # yield alice, bobs, results


    return StreamingHttpResponse(simulate_enhanced_communication(p_alice,p_bob,num_bobs,disconnect_percentage, message_length), content_type="application/json")



@csrf_exempt
def Alice_mnist(request):

    data = request.GET
    drop_probability = float(data.get('drop_probability', 0.6))
    num_samples = int(data.get('num_samples', 1))


  
    # Charger le modèle sauvegardé
    model = CNNAutoencoder()
    
    pth = settings.BASE_DIR / "static/data_pth/mnist_autoencoder.pth"
    model.load_state_dict(torch.load(pth))

    # model.load_state_dict(torch.load("mnist_autoencoder.pth"))
    model.eval()  # Mettre le modèle en mode évaluation

    # Définir la transformation pour les nouvelles données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float()),
    ])

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Créer un DataLoader pour itérer sur les données
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Fonction pour appliquer le dropout
    def random_drop_784(batch_images_flat, drop_probability=0.6):
        mask = (torch.rand_like(batch_images_flat) > drop_probability)
        dropped_flat = batch_images_flat.clone()
        dropped_flat[mask == 0] = -1
        return dropped_flat

    # Utiliser la fonction pour afficher les prédictions

    model = model.to(DEVICE)

    # Fonction pour afficher les prédictions
    def show_random_predictions(model, test_loader, DEVICE, num_samples=1, drop_probability=0.6):
        model.eval()
        
        # Charger toutes les données de test sous forme de liste
        all_data = list(test_loader)
        
        # Sélectionner un échantillon aléatoire parmi les données de test
        sample_images, _ = random.choice(all_data)
        
        # Limiter le nombre d'images affichées
        sample_images = sample_images[:num_samples].to(DEVICE)
        
        # Appliquer le dropout sur les images aplaties
        flat_sample = sample_images.view(num_samples, -1)
        dropped_flat_sample = random_drop_784(flat_sample, drop_probability=drop_probability)

        dropped_input_sample = dropped_flat_sample.view(num_samples, 1, 28, 28)
        
        # Obtenir les prédictions du modèle
        with torch.no_grad():
            logits = model(dropped_input_sample)
            reconstructed = torch.sigmoid(logits)
        
        # Convertir les tenseurs en CPU et en numpy pour l'affichage
        original_cpu = sample_images.cpu().numpy()
        dropped_cpu = dropped_input_sample.cpu().numpy()
        reconstructed_cpu = reconstructed.cpu().numpy()
        
        # Afficher les résultats avec matplotlib
        # plt.figure(figsize=(10, 5))
        for i in range(num_samples):

            yield json.dumps({"original":original_cpu[i][0].tolist(), "dropped":dropped_cpu[i][0].tolist(), "reconstructed":reconstructed_cpu[i][0].tolist()}) + "\n"
            time.sleep(1)


        
    return StreamingHttpResponse(show_random_predictions(model, test_loader, DEVICE, num_samples, drop_probability), content_type="application/json")

