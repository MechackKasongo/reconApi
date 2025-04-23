

from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io
from deepface import DeepFace

app = Flask(__name__)

# Charger le modèle entraîné
model_path = '/home/mechack-kasongo/Desktop/reconnaissance/model_VGG-Face.pkl'
try:
    model = joblib.load(model_path)
    print(f"✅ Modèle chargé depuis : {model_path}")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None

# Dictionnaire pour mapper les prédictions à des noms (à adapter selon tes classes)
label_to_name = {0: "dan_kileka", 1: "dido_mutombo", 2: "mechack_kasongo", 3: "guerschom_ngandu"}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image fournie'}), 400

    image_file = request.files['image']
    try:
        # Lire l'image et la convertir en tableau NumPy
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        img = img.resize((224, 224)) # Redimensionner à la taille attendue par VGG-Face (à vérifier)
        img_array = np.array(img)
        img_array = img_array / 255.0 # Normaliser les pixels

        # Extraction des embeddings avec DeepFace
        try:
            embedding_dict = DeepFace.represent(img_array, model_name="VGG-Face", enforce_detection=False) # Utilise le modèle que tu as entraîné
            if embedding_dict and embedding_dict[0] and "embedding" in embedding_dict[0]:
                embedding = np.array(embedding_dict[0]["embedding"]).reshape(1, -1)
            else:
                return jsonify({'error': 'Impossible d\'extraire l\'embedding du visage'}), 400
        except Exception as e:
            return jsonify({'error': f'Erreur lors de l\'extraction de l\'embedding : {e}'}), 400

        # Faire la prédiction
        prediction_proba = model.predict_proba(embedding)[0]
        predicted_class = np.argmax(prediction_proba)
        confidence = prediction_proba[predicted_class]

        predicted_name = label_to_name.get(predicted_class, 'Inconnu')

        result = {
            'predicted_class': int(predicted_class),
            'predicted_name': predicted_name,
            'confidence': float(confidence) * 100
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Erreur lors du traitement de l\'image : {e}'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')



# from flask import Flask, request, jsonify
# import joblib
# import numpy as np
# from PIL import Image
# import io
# from deepface import DeepFace

# app = Flask(__name__)

# # Charger le modèle entraîné
# model_path = '/home/mechack-kasongo/Desktop/reconnaissance/model_VGG-Face.pkl'
# try:
#     model = joblib.load(model_path)
#     print(f"✅ Modèle chargé depuis : {model_path}")
# except Exception as e:
#     print(f"❌ Erreur lors du chargement du modèle : {e}")
#     model = None

# # Dictionnaire pour mapper les prédictions à des noms (à adapter selon tes classes)
# label_to_name = {0: "dan_kileka", 1: "dido_mutombo", 2: "mechack_kasongo", 3: "guerschom_ngandu"}

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({'error': 'Modèle non chargé'}), 500

#     if 'image' not in request.files:
#         return jsonify({'error': 'Aucune image fournie'}), 400

#     image_file = request.files['image']
#     try:
#         # Lire l'image et la convertir en tableau NumPy
#         img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
#         img = img.resize((224, 224)) # Redimensionner à la taille attendue par VGG-Face (à vérifier)
#         img_array = np.array(img)
#         img_array = img_array / 255.0 # Normaliser les pixels

#         # Extraction des embeddings avec DeepFace
#         try:
#             embedding_dict = DeepFace.represent(img_array, model_name="VGG-Face", enforce_detection=False) # Utilise le modèle que tu as entraîné
#             if embedding_dict and embedding_dict[0] and "embedding" in embedding_dict[0]:
#                 embedding = np.array(embedding_dict[0]["embedding"]).reshape(1, -1)
#             else:
#                 return jsonify({'error': 'Impossible d\'extraire l\'embedding du visage'}), 400
#         except Exception as e:
#             return jsonify({'error': f'Erreur lors de l\'extraction de l\'embedding : {e}'}), 400

#         # Faire la prédiction
#         prediction_proba = model.predict_proba(embedding)[0]
#         predicted_class = np.argmax(prediction_proba)
#         confidence = prediction_proba[predicted_class]

#         predicted_name = label_to_name.get(predicted_class, 'Inconnu')

#         result = {
#             'predicted_class': int(predicted_class),
#             'predicted_name': predicted_name,
#             'confidence': float(confidence) * 100
#         }
#         return jsonify(result)

#     except Exception as e:
#         return jsonify({'error': f'Erreur lors du traitement de l\'image : {e}'}), 400

# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0')
