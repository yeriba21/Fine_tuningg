from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle
model = models.resnet50(pretrained=False)
num_classes = 113  # Remplacez par le nombre de classes dans votre dataset
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # Charger sur CPU
model.eval()  # Passer en mode évaluation

# Définir les transformations pour les images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Redimensionner les images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Endpoint pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    # Lire l'image
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Prétraiter l'image
    image = transform(image).unsqueeze(0)  # Ajouter une dimension de batch

    # Faire la prédiction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = predicted.item()

    # Renvoyer la prédiction
    return jsonify({'prediction': int(prediction)})

# Démarrer l'application Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)