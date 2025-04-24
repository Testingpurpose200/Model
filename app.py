from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import numpy as np
from flask_cors import CORS
import tensorflow.keras.backend as K
from keras.layers import DepthwiseConv2D
import logging

# 👇 This disables all GPUs and forces CPU usage
tf.config.set_visible_devices([], 'GPU')

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 5MB limit
CORS(app)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define custom focal loss function
def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    y_true = tf.cast(y_true, tf.float32)
    cross_entropy = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
    weights = alpha * tf.pow(1 - y_pred, gamma)
    loss = weights * cross_entropy
    return tf.reduce_mean(loss)

class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=None, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(*args, **kwargs)

# Load the model
MODEL_PATH = "model/model.h5"
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'DepthwiseConv2D': CustomDepthwiseConv2D,
        'focal_loss_fixed': focal_loss_fixed
    }
)
model.compile(optimizer=Adam(), loss=focal_loss_fixed, metrics=['accuracy'])

IMG_SIZE = (128, 128)

class_labels = [
    'aerosol_cans', 'aluminum_food_cans', 'battery', 'cardboard_boxes', 'clothes',
    'disposable_plastic_cutlery', 'eggshells', 'food_waste', 'glass_bottles',
    'glass_cosmetic_containers', 'glass_food_jars', 'magazines', 'paper', 'paper_cups',
    'plastic_detergent_bottles', 'plastic_food_containers', 'plastic_shopping_bags',
    'plastic_soda_bottles', 'plastic_straws', 'plastic_water_bottles', 'shoes',
    'styrofoam_food_containers'
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file type. Allowed: png, jpg, jpeg", 400

    try:
        img = Image.open(file).convert("RGB")
        img = img.resize(IMG_SIZE)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence_score = np.max(prediction)

        if confidence_score >= 0.5:
            class_label = class_labels[predicted_class]

            logging.debug("========== PREDICTION LOG ==========")
            logging.debug(f"Raw Model Output: {prediction}")
            logging.debug(f"Predicted Index: {predicted_class}")
            logging.debug(f"Predicted Label: {class_label}")
            logging.debug(f"Confidence Score: {confidence_score:.2f}")
            logging.debug("====================================")
        else:
            class_label = "Unknown (Confidence below 50%)"

        return jsonify({
            'prediction': class_label,
            'confidence': float(confidence_score)
        })

    except Exception as e:
        logging.exception("Error during prediction")
        return "Internal server error", 500

@app.errorhandler(413)
def file_too_large(e):
    return "File too large. Maximum size is 10MB.", 413

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses the PORT environment variable
    app.run(debug=True, host='0.0.0.0', port=port)
