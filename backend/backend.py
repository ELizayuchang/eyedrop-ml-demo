from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

model = load_model('model/model.h5')
class_indices = {'close eyes': 0, 'open eyes': 1}
index_to_class = {v: k for k, v in class_indices.items()}
img_size = (224, 224)

def prepare_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(img_size) 
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    img_bytes = file.read()
    img_tensor = prepare_image(img_bytes)
    pred = model.predict(img_tensor)
    pred_class = np.argmax(pred, axis=1)[0]
    class_name = index_to_class[pred_class]
    return jsonify({'prediction': class_name})

@app.route('/')
def serve_index():
    return send_from_directory('../frontend', 'index.html')

if __name__ == '__main__':
    app.run(debug=True,port = 8080)