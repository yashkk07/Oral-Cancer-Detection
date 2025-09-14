from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the trained Keras model
model = load_model("oralcancern.keras")

# Define class labels for binary classification
labels = ['Normal', 'OSCC']

@app.route('/')
def index():
    # Optionally, return a web interface (index.html) or remove this if API-only
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    filepath = os.path.join("static", "uploaded_image.jpg")
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(128, 128))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = img_tensor / 255.0  # Normalize

    # Predict
    probability = model.predict(img_tensor)[0][0]  # Output is a single sigmoid value

    # Apply threshold
    threshold = 0.42  # Change this if you've tuned a better threshold
    predicted_class = labels[int(probability > threshold)]
    confidence = float(probability if probability > threshold else 1 - probability)

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence, 3),
        'probability': round(float(probability), 4),
        'raw_output': float(probability)
    })

if __name__ == '__main__':
    app.run(debug=True)
