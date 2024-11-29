from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Define the path to the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your model
model_path = 'prediction_medicinal.h5'  # Change this to your model's path
model = load_model(model_path)

def predict_stage(img_path):
    # Load the image you want to predict
    img = image.load_img(img_path, target_size=(400, 400))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.  # Rescale pixel values to [0, 1]

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Map predicted class index to class label
    class_labels = ['Arive-Dantu','Basale','Betel','Crape_Jasmine','Curry','Drumstick','Fenugreek','Guava','Hibiscus','Indian_Beech','Indian_Mustard','Jackfruit','Jamaica_Cherry-Gasagase','Jamun','Jasmine','Karanda','Lemon','Mango','Mexican_Mint','Mint','Neem','Oleander','Parijata','Peepal','Pomegranate','Rasna','Rose_apple','Roxburgh_fig','Sandalwood','Tulsi']

    predicted_label = class_labels[predicted_class]
    prediction_probabilities = [round(prob, 2) for prob in prediction[0]]

    return predicted_label, prediction_probabilities

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')
        
        file = request.files['file']

        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            # Save the file to the uploads folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict the image class
            predicted_class, prediction_probabilities = predict_stage(file_path)

            # Render the result template with the image and prediction
            return render_template('result.html', image_file=file_path, predicted_class=predicted_class, probabilities=prediction_probabilities)

if __name__ == '__main__':
    app.run(debug=True)
