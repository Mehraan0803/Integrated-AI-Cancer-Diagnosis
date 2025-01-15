from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import cv2
import os

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = load_model('uairesnet50.h5')

# Define labels for predictions
all_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor', 'cancer', 'no_cancer', 'Healthy', 'Sick']

# Map model labels to human-readable labels
label_map = {
    'glioma_tumor': 'Glioma Tumor',
    'meningioma_tumor': 'Meningioma Tumor',
    'no_tumor': 'No Brain Tumor',
    'pituitary_tumor': 'Pituitary Tumor',
    'cancer': 'Lung Cancer',
    'no_cancer': 'No Lung Cancer',
    'Healthy': 'No Breast Cancer',
    'Sick': 'Breast Cancer'
}

# Ensure the 'static/uploads' folder exists
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for the homepage (index)
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction after image upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file to static/uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Open the uploaded image properly (make sure it's read as binary)
    img = cv2.imread(file_path)
    
    # If the image is invalid (None), return an error
    if img is None:
        return "Unable to read the uploaded image. Please upload a valid image file.", 400

    # Preprocess the uploaded image
    img = cv2.resize(img, (150, 150))  # Resize to match the ResNet50 input size (224x224)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess image (for ResNet50)

    # Get the model's predictions
    predictions = model.predict(img)
    predicted_label = all_labels[np.argmax(predictions)]  # Get the class with the highest probability

    # Map the predicted label to a human-readable format
    readable_prediction = label_map.get(predicted_label, predicted_label)

    # Return the result and show the uploaded image
    return render_template('index.html', prediction=readable_prediction, img_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)




