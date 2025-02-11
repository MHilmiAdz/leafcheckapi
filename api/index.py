from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import traceback  # Import traceback for better error logs

# Initialize the Flask application
app = Flask(__name__)
model = load_model('api\model.h5')  # Load the pre-trained model
app.secret_key = 'supersecretkey'  # Necessary for flash messages
# Set the upload folder and allowed file extensions
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    
@app.route('/')
def index_view():
    # Render the index page
    return {"msg":"API READY"}

def allowed_file(filename):
    # Check if the uploaded file has one of the allowed extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path, target_size=(224, 224)):
    # Load, resize, convert to array, and preprocess the image
    img = load_img(file_path, target_size=target_size)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return {"error": "No file part"}

        file = request.files['file']

        if file.filename == '':
            return {"error": "No file selected"}

        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure the upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            file.save(file_path)

            # Process the image
            img = load_img(file_path, target_size=(150, 150))
            x = img_to_array(img)
            x /= 255.0
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            # Debug: Print image shape
            print("Image shape:", x.shape)

            # Make the prediction
            classes = model.predict(images, batch_size=10)
            predicted_class = np.argmax(classes[0])

            # Debug: Print model prediction
            print("Model prediction:", classes)
            print("Predicted class:", predicted_class)

            # Map prediction to labels
            label_map = {
                0: (1, "healthy apple leaves", "Daun buah apel yang sehat berwarna hijau cerah, mengilap, dan bebas dari bintik-bintik atau lubang. Bentuk daun biasanya oval dengan tepi bergerigi dan urat-urat daun terlihat jelas. Daun terasa kuat dan elastis saat disentuh."),
                1: (1, "unhealthy apple leaves", "Daun buah apel yang tidak sehat sering kali berwarna kuning, cokelat, atau memiliki bintik-bintik hitam. Daun bisa tampak layu, kusam, dan mungkin memiliki lubang akibat serangan hama atau penyakit. Tekstur daun biasanya kering atau rapuh"),
                2: (2, "healthy mango leaves", "Daun mangga yang sehat memiliki warna hijau tua dan mengilap, dengan urat daun yang terlihat jelas. Bentuknya lonjong dan ujung meruncing, dengan permukaan yang halus dan tekstur yang kuat serta elastis."),
                3: (2, "unhealthy mango leaves", "Daun mangga yang tidak sehat sering kali berwarna kuning, cokelat, atau terdapat bintik-bintik hitam. Daun mungkin terlihat keriput, layu, atau berlubang akibat hama. Tekstur daun biasanya kering dan mudah patah."),
                4: (3, "healthy orange leaves", "Daun jeruk yang sehat berwarna hijau tua, mengilap, dan memiliki tekstur yang halus. Daun berbentuk oval atau lonjong dengan ujung meruncing dan urat-urat daun terlihat jelas. Daun terasa kenyal saat disentuh dan bebas dari bintik atau kerusakan"),
                5: (3, "unhealthy orange leaves", "Daun jeruk yang tidak sehat biasanya berwarna kuning, cokelat, atau memiliki bercak-bercak hitam. Daun dapat terlihat layu, keriput, atau berlubang akibat hama. Tekstur daun kering dan rapuh"),
            }

            leaftype, label, notes = label_map.get(predicted_class, ("Unknown", "Data tidak ditemukan"))

            # Debug: Print response
            response = {
                "status": "success",
                "leaftype": leaftype,
                "leaf": label,
                "user_image": filename,
                "keterangan": notes,
            }
            print("Response:", response)

            return response

        return {
            "status": "400",
            "error": "Invalid file type"}

    except Exception as e:
        print(traceback.format_exc())  # Print full error traceback
        return {"error": "Internal Server Error"}

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port='8000')