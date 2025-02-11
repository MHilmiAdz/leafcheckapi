from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import traceback

# Initialize Flask app
app = Flask(__name__)
model = load_model('model/model.h5')  # Load the trained model

UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure upload folder exists

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Label mapping
LABEL_MAP = {
    0: (1, "healthy apple leaves", "Daun buah apel yang sehat berwarna hijau cerah, mengilap, dan bebas dari bintik-bintik atau lubang."),
    1: (1, "unhealthy apple leaves", "Daun buah apel yang tidak sehat sering kali berwarna kuning, cokelat, atau memiliki bintik-bintik hitam."),
    2: (2, "healthy mango leaves", "Daun mangga yang sehat memiliki warna hijau tua dan mengilap, dengan urat daun yang terlihat jelas."),
    3: (2, "unhealthy mango leaves", "Daun mangga yang tidak sehat sering kali berwarna kuning, cokelat, atau terdapat bintik-bintik hitam."),
    4: (3, "healthy orange leaves", "Daun jeruk yang sehat berwarna hijau tua, mengilap, dan memiliki tekstur yang halus."),
    5: (3, "unhealthy orange leaves", "Daun jeruk yang tidak sehat biasanya berwarna kuning, cokelat, atau memiliki bercak-bercak hitam."),
}

@app.route('/')
def index():
    return jsonify({"msg": "API READY"})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path, target_size=(150, 150)):
    img = Image.open(file_path).resize(target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        if not file or not allowed_file(file.filename):
            return jsonify({"error": "Invalid or missing file"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Process image
        image_array = preprocess_image(file_path)
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction[0])

        # Get label info
        leaftype, label, description = LABEL_MAP.get(predicted_class, ("Unknown", "Data tidak ditemukan"))
        
        return jsonify({
            "status": "success",
            "leaftype": leaftype,
            "leaf": label,
            "user_image": file.filename,
            "keterangan": description
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run()
