from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load model only once at startup to avoid repeated loading
model = tf.keras.models.load_model("my_model.keras")

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict-form')
def predict_form():
    return render_template('predict_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Predict class
        result_index = model_prediction(file_path)
        predicted_class = CLASS_NAMES[result_index]

        # Clean up file
        os.remove(file_path)

        return jsonify({"Predicted Disease": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def model_prediction(file_path):
    # Load and preprocess image
    image = tf.keras.preprocessing.image.load_img(file_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

if __name__ == "__main__":
    app.run()
