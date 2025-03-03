from django.shortcuts import render

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import io
import base64
import numpy as np
from django.shortcuts import render

# Configuration du modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class CNNAutoencoder(nn.Module):
#     def __init__(self):
#         super(CNNAutoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
#         )
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

# # Charger le modèle entraîné
# model = CNNAutoencoder().to(device)
# model.load_state_dict(torch.load("mnist_autoencoder.pth", map_location=device))
# model.eval()

# # Transformation des images
# def binarize(tensor):
#     return (tensor > 0.5).float()

# def preprocess_image(image):
#     transform = transforms.Compose([
#         transforms.Grayscale(num_output_channels=1),
#         transforms.Resize((28, 28)),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: binarize(x)),
#     ])
#     return transform(image).unsqueeze(0).to(device)  # Ajouter une dimension batch

# @csrf_exempt
# def predict(request):
#     if request.method == 'POST':
#         try:
#             image_data = request.FILES['image'].read()
#             image = Image.open(io.BytesIO(image_data)).convert('L')
#             processed_image = preprocess_image(image)
            
#             # Passer l'image dans l'autoencodeur
#             with torch.no_grad():
#                 output = model(processed_image)
#                 output = torch.sigmoid(output).cpu().numpy()[0, 0]  # Convertir en numpy
            
#             # Convertir l'image en base64 pour l'affichage
#             output_image = Image.fromarray((output * 255).astype(np.uint8))
#             buffered = io.BytesIO()
#             output_image.save(buffered, format="PNG")
#             img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
#             return JsonResponse({'reconstructed_image': img_str})
#         except Exception as e:
#             return JsonResponse({'error': str(e)}, status=400)
#     return JsonResponse({'message': 'Use POST request to send an image.'}, status=400)





def index(request):
    return render(request, 'index.html')