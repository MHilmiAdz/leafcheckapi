import os
import logging
import traceback
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS # type: ignore
from dotenv import load_dotenv # type: ignore
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from PIL import Image

# Load environment variables
load_dotenv()
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/images")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Flask App Configuration
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size
CORS(app)  # Enable Cross-Origin Resource Sharing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_model():
    if "model" not in g:
        g.model = load_model('model\model.h5')
    return g.model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    img = Image.open(file_path).resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/')
def index_view():
    return {"msg": "API READY"}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        x = preprocess_image(file_path)
        model = get_model()
        prediction = model.predict(x)
        predicted_class = np.argmax(prediction[0])

        label_map = {
                0: (1, "healthy apple leaves", "Daun buah apel yang sehat berwarna hijau cerah, mengilap, dan bebas dari bintik-bintik atau lubang. Bentuk daun biasanya oval dengan tepi bergerigi dan urat-urat daun terlihat jelas. Daun terasa kuat dan elastis saat disentuh."),
                1: (1, "unhealthy apple leaves", "Daun buah apel yang tidak sehat sering kali berwarna kuning, cokelat, atau memiliki bintik-bintik hitam. Daun bisa tampak layu, kusam, dan mungkin memiliki lubang akibat serangan hama atau penyakit. Tekstur daun biasanya kering atau rapuh"),
                2: (2, "healthy mango leaves", "Daun mangga yang sehat memiliki warna hijau tua dan mengilap, dengan urat daun yang terlihat jelas. Bentuknya lonjong dan ujung meruncing, dengan permukaan yang halus dan tekstur yang kuat serta elastis."),
                3: (2, "unhealthy mango leaves", "Daun mangga yang tidak sehat sering kali berwarna kuning, cokelat, atau terdapat bintik-bintik hitam. Daun mungkin terlihat keriput, layu, atau berlubang akibat hama. Tekstur daun biasanya kering dan mudah patah."),
                4: (3, "healthy orange leaves", "Daun jeruk yang sehat berwarna hijau tua, mengilap, dan memiliki tekstur yang halus. Daun berbentuk oval atau lonjong dengan ujung meruncing dan urat-urat daun terlihat jelas. Daun terasa kenyal saat disentuh dan bebas dari bintik atau kerusakan"),
                5: (3, "unhealthy orange leaves", "Daun jeruk yang tidak sehat biasanya berwarna kuning, cokelat, atau memiliki bercak-bercak hitam. Daun dapat terlihat layu, keriput, atau berlubang akibat hama. Tekstur daun kering dan rapuh"),
            }

        leaftype, label, notes = label_map.get(predicted_class, ("Unknown", "Data not found"))
        response = {"status": "success", "leaftype": leaftype, "leaf": label, "user_image": filename, "notes": notes}
        logger.info(f"Response: {response}")

        return jsonify(response)

    except Exception as e:
        logger.error("Error processing request", exc_info=True)
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run()