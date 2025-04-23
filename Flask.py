from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
import os
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)

# Define constants
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
IM_HEIGHT = 256
IM_WIDTH = 256

# Configure upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Create folders if not exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Define custom metrics for model loading
def dice_coefficients(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

def dice_coefficients_loss(y_true, y_pred, smooth=100):
    return -dice_coefficients(y_true, y_pred, smooth)

def iou(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum_union = K.sum(y_true + y_pred)
    return (intersection + smooth) / (sum_union - intersection + smooth)

# Load the pre-trained model
model = load_model("unet_brain_mri_seg.hdf5", custom_objects={
    'dice_coef_loss': dice_coefficients_loss, 'iou': iou, 'dice_coef': dice_coefficients
})

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded!")

        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return render_template('index.html', error="No file selected!")

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process the image
        try:
            image = cv2.imread(filepath)
            image_resized = cv2.resize(image, (IM_HEIGHT, IM_WIDTH))
            image_normalized = image_resized / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)

            # Predict the segmented image
            prediction = model.predict(image_input)
            prediction_image = (prediction[0, :, :, 0] * 255).astype(np.uint8)
            prediction_filepath = os.path.join(app.config['RESULT_FOLDER'], f"result_{filename}")
            cv2.imwrite(prediction_filepath, prediction_image)

            return render_template('result.html', original_image=filepath, result_image=prediction_filepath)
        except Exception as e:
            return render_template('index.html', error=f"Error processing file: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
