from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the Keras model
model = load_model("kan_oralcancer_model1.keras")

# Define class labels if binary classification
labels = ['Normal', 'OSCC']

@app.route('/')
def index():
    return render_template('index.html')  # or just return a JSON if API-only

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
    img_tensor /= 255.0

    # Predict using softmax
    prediction = model.predict(img_tensor)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = labels[predicted_class_index]
    confidence = float(prediction[0][predicted_class_index])

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence, 3),
        'raw_output': prediction[0].tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
