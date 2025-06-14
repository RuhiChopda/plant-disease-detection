import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, make_response, redirect

# Initialize app
app = Flask(__name__)
MODEL_PATH = 'model/plant_disease_model.h5'
LABELS_PATH = 'model/labels.npy'

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and labels
model = load_model(MODEL_PATH)
class_names = np.load(LABELS_PATH)

# Risk levels
risk_levels = {
    "healthy": {"en": "No Risk", "hi": "कोई जोखिम नहीं"},
    "Early_blight": {"en": "Medium Risk", "hi": "मध्यम जोखिम"},
    "Late_blight": {"en": "High Risk", "hi": "उच्च जोखिम"},
    "Leaf_Mold": {"en": "Medium Risk", "hi": "मध्यम जोखिम"},
    "Septoria_leaf_spot": {"en": "High Risk", "hi": "उच्च जोखिम"},
    "Bacterial_spot": {"en": "High Risk", "hi": "उच्च जोखिम"},
    "Target_Spot": {"en": "Medium Risk", "hi": "मध्यम जोखिम"}
}

# Language logic
def get_language():
    return 'hi' if request.cookies.get('language') == 'hi' else 'en'

# Set language route (optional, to dynamically change language)
@app.route('/set_language/<lang>', methods=['GET'])
def set_language(lang):
    resp = make_response(redirect(request.referrer))  # Redirect back to the previous page
    resp.set_cookie('language', lang)
    return resp

# Home route
@app.route('/')
def home():
    language = get_language()  # Get language from cookies or default to 'en'
    return render_template('index.html', language=language)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(64, 64))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    # Find risk level
    risk = "Unknown"
    for key in risk_levels:
        if key in predicted_class:
            risk = risk_levels[key][get_language()]
            break
    if "healthy" in predicted_class:
        risk = risk_levels["healthy"][get_language()]

    # Get path for displaying
    display_path = os.path.join('uploads', file.filename)

    language = get_language()  # Get language from cookies or default to 'en'
    return render_template('result.html', prediction=predicted_class, risk=risk, image_path=display_path, language=language)

if __name__ == "__main__":
    # Make sure uploads folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
