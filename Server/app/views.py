from django.shortcuts import render
from django.http import StreamingHttpResponse
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import time
import json
from django.conf import settings

from .outils.Node import Node, Alice , Node2
from .outils.CNNAutoencoder import CNNAutoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def index(request):
    return render(request, 'index.html')

def Alice_Bob_2(p_alice, p_bob, message_length, disconnect_percentage):
    Node.reset()
    alice = Node(node_id=1, p_one_value=p_alice, learning_rate=0.1)
    bob = Node(node_id=2, p_one_value=p_bob, learning_rate=0.1)



    # Connecter les nœuds
    Node.connect_nodes([(1, 2)])


    num_turns = 1000
    for turn in range(num_turns):
        # print(f"\n--- Tour {turn + 1} ---")
        Node.turn()


    true = []
    predicted = []

    for step in range(1, message_length + 1):
        bit_from_alice = alice.send_message()
        bit_from_bob = bob.send_message()
        disconnected = step >= message_length * disconnect_percentage / 100

        guess = "."
        if disconnected :
            guess = alice.guess_node_bit(bob.node_id)
            true.append(bit_from_bob)
            predicted.append(guess)


   
        else:
            alice.receive_message(bit_from_bob,bob.node_id)
            bob.receive_message(bit_from_alice,alice.node_id)

        message = json.dumps({"step": step,"alice":bit_from_alice,"bob":bit_from_bob ,"guess":guess, "disconnected":disconnected}) + "\n"
        yield message
        # time.sleep(1)
    for node in [alice, bob]:
        print(f"\nNœud {node.node_id} - Comparaison des p-values:")
        for other_node_id, estimated_p in node.p_values.items():
            true_p = Node.get_node(other_node_id).p_one_value
            print(f"  Nœud {other_node_id} - Estimé: {estimated_p:.4f}, Vrai: {true_p:.4f}, Erreur: {abs(estimated_p - true_p):.4f}")

    print(true , sum(true))
    print(predicted , sum(predicted))
    correct_predictions = sum(1 for real, predicted in zip(true, predicted) if real == predicted)
    accuracy = (correct_predictions / message_length) * 100
    print(f"Prédictions correctes: {correct_predictions} / {message_length} ({accuracy:.2f}%)")

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
            alice.learn()
            bob.learn()
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


    return StreamingHttpResponse(Alice_Bob_2(p_alice,p_bob,message_length,disconnect_percentage), content_type="application/json")
    return StreamingHttpResponse(Alice_Bob(p_alice,p_bob,message_length,disconnect_percentage), content_type="application/json")

def simulate_enhanced_communication_2(p_alice, p_bob, num_bobs, disconnect_percentage, message_length):
    """
    Lance une simulation améliorée avec plusieurs Bobs.
    """
    Node.reset()
    alice = Node(node_id=1, p_one_value=p_alice, learning_rate=0.1)
    bobs = [Node(node_id=i+2, p_one_value=random.random(),learning_rate=0.1) for i in range(num_bobs)]

    # Connecter les nœuds
    Node.connect_nodes([(1, bob.node_id) for bob in bobs])



    num_disconnected = int(num_bobs * disconnect_percentage / 100)
    disconnected_bobs = random.sample(range(1,num_bobs+1), num_disconnected)
    disconnect_steps = {bob_id: random.randint(1, message_length-1) for bob_id in disconnected_bobs} # Étapes de déconnexion


    for step in range(1, message_length + 1):
        bit_from_alices = 0
        bit_from_bobs = {}
        would_send_bits = {}
        disconnecteds = []

        bit_from_alice = alice.send_message()
        for bob in bobs:

            
            would_send_bits[bob.node_id] = alice.guess_node_bit(bob.node_id)

            bit_from_bob = bob.send_message()
            bit_from_bobs[bob.node_id] = bit_from_bob

            if bob.node_id in disconnected_bobs and step == disconnect_steps[bob.node_id]:
                disconnecteds.append(bob.node_id)

            if bob.node_id not in disconnected_bobs or step < disconnect_steps[bob.node_id]:
    
                alice.receive_message(bit_from_bob,bob.node_id)
                bob.receive_message(bit_from_alice,alice.node_id)
        alice.learn()

        message = json.dumps({"step": step , "alice":bit_from_alices, "bob":bit_from_bobs, "would_send":would_send_bits,"disconnecteds":disconnecteds}) + "\n"
        yield message
        time.sleep(1)

    # for node in [alice] + bobs:
    for node in [alice] :
        print(f"\nNœud {node.node_id} - Comparaison des p-values:")
        for other_node_id, estimated_p in node.p_values.items():
            true_p = Node.get_node(other_node_id).p_one_value
            print(f"  Nœud {other_node_id} - Estimé: {estimated_p:.4f}, Vrai: {true_p:.4f}, Erreur: {abs(estimated_p - true_p):.4f}")   


def simulate_enhanced_communication(p_alice, p_bob, num_bobs, disconnect_percentage, message_length):
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


@csrf_exempt
def AliceNBob(request):

    # get the data from the request
    data = request.GET
    p_alice = float(data.get('p_alice', 0.5))
    p_bob = float(data.get('p_bob', 0.5))
    num_bobs = int(data.get('nb_bob', 100))
    disconnect_percentage = int(data.get('disconnect_percentage', 20))
    message_length = int(data.get('message_length', 100))

    return StreamingHttpResponse(simulate_enhanced_communication_2(p_alice,p_bob,num_bobs,disconnect_percentage, message_length), content_type="application/json")



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



def can_join_mis(node_key, neighbors, mis):
    return all(neighbor not in mis for neighbor in neighbors[node_key])




def simulation_mis(nodes,connextions ,start_with_most_neighbors=False):
    """
    Lance une simulation de communication multi-nœuds.
    """
    Node.reset()
    nodes = [Node(node_id=i+1, p_one_value=random.random(), learning_rate=0.1) for i in range(nodes)]

    print(connextions)
    # Connecter les nœuds
    Node.connect_nodes(connextions)

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

        
    