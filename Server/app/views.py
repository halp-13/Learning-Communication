from django.shortcuts import render
from django.http import StreamingHttpResponse
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from django.views.decorators.csrf import csrf_exempt
import time
import json
from django.conf import settings

from .outils.Node import Node, Alice , Node_Mis
from .outils.CNNAutoencoder import CNNAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def index(request):
    return render(request, 'index.html')


###############################################################
# Fonctions pour les simulations de communication alice-bob
###############################################################


def Alice_Bob(p_alice, p_bob, message_length, disconnect_percentage):
    
    num_disconnected = int(message_length * disconnect_percentage / 100)

    alice = Node("Alice", p_alice)
    bob = Node("Bob", p_bob )

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
        
        distance = np.linalg.norm(bit_from_alice - bit_from_bob)
        message = json.dumps({"step": step,"alice":bit_from_alice,"bob":bit_from_bob ,"guess":guess, "disconnected":disconnected ,"distance":distance }) + "\n"
        yield message
        time.sleep(1)



    correct_predictions = sum(1 for real, predicted in zip(results["real_bits"], results["predicted_bits"]) if real == predicted)
    accuracy = (correct_predictions / message_length) * 100

    distance = 0 #np.linalg.norm(np.array(results["real_bits"]) - np.array(results["predicted_bits"]))

    yield json.dumps({"correct_predictions":correct_predictions, "accuracy":accuracy ,"distance":distance}) + "\n"


@csrf_exempt
def AliceBob(request):
    "Just 2 Nodes Communication (alice and bob) afeter bob disconnected alice will guess the message"
    # mode = 'randomn', ,message=''
    data = request.GET
    p_alice = float(data.get('p_alice', 0.5))
    p_bob = float(data.get('p_bob', 0.5))
    message_length = int(data.get('message_length', 100))
    disconnect_percentage = int(data.get('disconnect_percentage', 20))


    return StreamingHttpResponse(Alice_Bob(p_alice,p_bob,message_length,disconnect_percentage), content_type="application/json")
    return StreamingHttpResponse(Alice_Bob(p_alice,p_bob,message_length,disconnect_percentage), content_type="application/json")

###############################################################
# Fonctions pour les simulations de communication alice-n-bob
###############################################################




def simulate_enhanced_communication(p_alice=0.5, p_bob=0.5, num_bobs=100, disconnect_percentage=20, message_length=100, buffer_size=40, model_type="random_forest"):
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
        bit_from_alices = {}
        bit_from_bobs = {}
        would_send_bits = {}
        disconnecteds = []
        if step % 10 == 0 and step > 20:
            alice.train_model()
        
        for bob_id, bob in enumerate(bobs):
            if bob_id in disconnected_bobs and step == disconnect_steps[bob_id] and not bob.is_disconnected:
                bob.is_disconnected = True
                disconnecteds.append(bob_id)
            
            bit_from_alice = alice.send_message()

            if bit_from_alice is not None and not bob.is_disconnected:
                bob.receive_message(bit_from_alice)
                bit_from_alices[bob_id] = bit_from_alice
            
            would_send_bit = 1 if random.random() < p_bob else 0
            would_send_bits[bob_id] = would_send_bit

            
            results["would_send_bits"][bob_id].append(would_send_bit)
            
            
            bit_from_bob = bob.send_message()
            bit_from_bobs[bob_id] = bit_from_bob
            
            if bit_from_bob is not None:
                alice.receive_message_from_bob(bob_id, bit_from_bob)
                results["real_bits"][bob_id].append(bit_from_bob)

            else:
                predicted_bit = alice.predict_message(bob_id)
                results["predicted_bits"][bob_id].append(predicted_bit)
    
        message = json.dumps({"step": step , "alice":bit_from_alices, "bob":bit_from_bobs, "would_send":would_send_bits,"disconnecteds":disconnecteds}) + "\n"
        yield message
        # time.sleep(1)
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
            
            #Cette partie du code permet d’afficher les détails pour chaque Bob.  
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



@csrf_exempt
def AliceNBob(request):

    # get the data from the request
    data = request.GET
    p_alice = float(data.get('p_alice', 0.5))
    p_bob = float(data.get('p_bob', 0.5))
    num_bobs = int(data.get('nb_bob', 100))
    disconnect_percentage = int(data.get('disconnect_percentage', 20))
    message_length = int(data.get('message_length', 100))

    return StreamingHttpResponse(simulate_enhanced_communication(p_alice,p_bob,num_bobs,disconnect_percentage, message_length), content_type="application/json")


###############################################################
# Fonctions pour les simulations de communication MNIST
###############################################################

# Charger le modèle sauvegardé
model = CNNAutoencoder()

pth = settings.BASE_DIR / "static/data_pth/mnist_autoencoder.pth"
model.load_state_dict(torch.load(pth))

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

# Charger toutes les données de test sous forme de liste

all_data = list(test_loader)

# Fonction pour afficher les prédictions
def show_random_predictions(model, all_data, DEVICE, num_samples, drop_probability):
    model.eval()
    
    
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
    for i in range(num_samples):
        yield json.dumps({"original":original_cpu[i][0].tolist(), "dropped":dropped_cpu[i][0].tolist(), "reconstructed":reconstructed_cpu[i][0].tolist()}) + "\n"
        time.sleep(1)

@csrf_exempt
def Alice_mnist(request):

    data = request.GET
    drop_probability = float(data.get('drop_probability', 0.6))
    num_samples = int(data.get('num_samples', 1))
        
    return StreamingHttpResponse(show_random_predictions(model, all_data, DEVICE, num_samples, drop_probability), content_type="application/json")

###############################################################
# Fonctions pour les simulations de communication mis
###############################################################

def can_join_mis(node_key, neighbors, mis):
    return all(neighbor not in mis for neighbor in neighbors[node_key])




def simulation_mis(nodes,connextions ,start_with_most_neighbors=False):
    """
    Lance une simulation de communication multi-nœuds.
    """
    Node_Mis.reset()
    nodes = [Node_Mis(node_id=i+1, p_one_value=random.random(), learning_rate=0.1) for i in range(nodes)]

    print(connextions)
    # Connecter les nœuds
    Node_Mis.connect_nodes(connextions)

    mis = set()
    neighbors = {node.node_id: set() for node in nodes}
    
    for edge in connextions:
        neighbors[edge[0]].add(edge[1])
        neighbors[edge[1]].add(edge[0])

    remaining_nodes = {node.node_id for node in nodes}
    

    cpt= 0
    while remaining_nodes:
        messages = {node.node_id: 0 if random.random() < node.p_one_value else 1 for node in nodes if node.node_id in remaining_nodes}
        
        for node_key in list(remaining_nodes):
            if messages.get(node_key) == 1:
                if all(messages.get(neighbor, 0) == 0 for neighbor in neighbors[node_key]) and can_join_mis(node_key, neighbors, mis):
                    mis.add(node_key)
        
        for node_key in mis:
            remaining_nodes.discard(node_key) # Supprimer les nœuds déjà dans l'ensemble indépendant maximal
            remaining_nodes.difference_update(neighbors[node_key]) # Supprimer les voisins des nœuds déjà dans l'ensemble indépendant maximal

        yield json.dumps({"mis":list(mis), "remaining_nodes":list(remaining_nodes), "messages":messages}) + "\n"
        time.sleep(1)
    

    # mis = maximal_independent_set(nodes, connextions, start_with_most_neighbors=True)
    
    print(f"Ensemble indépendant maximal: {mis}")
    


@csrf_exempt
def mis(request):
    data = json.loads(request.body)  # Récupérer et parser les données JSON
    nodes = data.get('nb_nodes', [])
    edges = data.get('edges', [])
    connextions = data.get('connextions', [])
    nb_nodes = int(data.get('nb_nodes', None))

    connextions = data.get('connextions', None)


    if nb_nodes != None and connextions !=[]:

        return StreamingHttpResponse(simulation_mis(nb_nodes,connextions), content_type="application/json")

        
    