from flask import Flask, request, jsonify
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, firestore
import os
import numpy as np;
import string
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Set credentials for Firestore (adjust the path to the file as needed)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "career-match-443816-51cc7628ccd0.json"

# Load TFLite model using TensorFlow
MODEL_PATH = r"saved_model/saved_model.pb"  # Ensure the model file is in the same directory
interpreter = None

try:
    model = tf.saved_model.load(r"saved_model")
    predict_fn = model.signatures['serving_default']
    print("Model loaded successfully.")
    # print(model.input_shape)
except Exception as e:
    print(f"Error loading model: {e}")
    interpreter = None

# Initialize Firestore
db = None
try:
    cred = credentials.Certificate(r"career-match-443816-51cc7628ccd0.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Connected to Firestore.")
except Exception as e:
    print(f"Error connecting to Firestore: {e}")
    db = None

@app.route("/")
def home():
    return {"message": "Welcome to Career Recommendation API"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure model is loaded
        if model is None:
            return jsonify({"error": "Model is not loaded."}), 500

        # Get user_input from request body
        data = request.get_json()
        if not data or "user_input" not in data:
            return jsonify({"error": "Invalid input."}), 400

        user_input = data["user_input"]

        job_categories = ['Sales and Marketing', 'Finance and Business Strategy', 'Software Development and IT Services', 'Hardware Engineering and Infrastructure', 'Operations and Support', 'Design and User Experience', 'Legal and Communications', 'Other']
        # Create a LabelEncoder instance
        label_encoder = LabelEncoder()
        label_encoder.fit(job_categories)
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        job_descriptions = [
            "shape shepherd shepherd ship ship show show technical technical programs programs designed designed support support work work cloud cloud customer [('experience business', 1), ('business technology', 3), ('technology market', 1), ('market program', 1), ('program manager', 1), ('manager saas', 1), ('saas cloud', 2), ('cloud computing', 3), ('computing and/or', 1), ('and/or emerging', 1)]",
            "drive cross-functional cross-functional activities activities supply supply chain chain overall overall technical technical operational operational readiness readiness npi npi phases [('bsee bsme', 1), ('bsme bsie', 1), ('bsie degree', 1), ('degree experience', 9), ('experience using', 11), ('using statistics', 1), ('statistics tools', 1), ('tools data', 1), ('data analysis', 5), ('analysis e.g', 1)]",
            "collect analyze analyze data data draw draw insight insight identify identify strategic strategic solutions solutions build build consensus consensus facilitating [('experience partnering', 1), ('partnering consulting', 1), ('consulting cross-functionally', 1), ('cross-functionally senior', 1), ('senior stakeholders', 1), ('stakeholders proficiency', 1), ('proficiency database', 1), ('database query', 1), ('query language', 1), ('language e.g', 1)]",
            "work one-on-one one-on-one top top android android ios ios web web engineers engineers build build exciting exciting new new product/api [('experience software', 7), ('software developer', 5), ('developer architect', 5), ('architect technology', 4), ('technology advocate', 3), ('advocate cto', 3), ('cto consultant', 3), ('consultant working', 5), ('working web', 3), ('web mobile', 3)]",
            "shape shepherd shepherd ship ship show show technical technical programs programs designed designed support support work work cloud cloud customer [('experience business', 1), ('business technology', 3), ('technology market', 1), ('market program', 1), ('program manager', 1), ('manager saas', 1), ('saas cloud', 2), ('cloud computing', 3), ('computing and/or', 1), ('and/or emerging', 1)]"
        ]

        # Fit the TF-IDF vectorizer to the job descriptions
        tfidf_vectorizer.fit(job_descriptions)

        text = predict_job_category(user_input,tfidf_vectorizer,model,label_encoder,job_categories)

        doc_ref = db.collection("predictions").document()
        doc_ref.set({
            "user_input": user_input,
            "prediction": text
        })

        print(text)

        return jsonify({"prediction": text})

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500
@app.route('/register', methods=['POST'])
def register():
    # Parse the incoming JSON request
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    phone_number = data.get('phone_number')

    # Input validation
    if not username or not email or not password or not phone_number:
        return {
            "error": True,
            "message": "Username, email, and password are required."
        }, int(400)

    # References to the Firestore documents
    user_ref = db.collection('users').document(username)
    email_ref = db.collection('users').document(email)
    counter_ref = db.collection('metadata').document('user_counter')

    try:
        # Check if the user already exists
        if user_ref.get().exists or email_ref.get().exists:
            return {
                "error": True,
                "message": f"User {username} already exists."
            }, int(400)

        # Increment the userID counter
        counter_snapshot = counter_ref.get()
        if counter_snapshot.exists:
            current_id = counter_snapshot.to_dict().get('latest_id', 0)
            new_id = current_id + 1
        else:
            new_id = 1  # Start at 1 if the counter doesn't exist

        # Update the counter document
        counter_ref.set({'latest_id': new_id}, merge=True)

        # Create a new user document
        user_ref.set({
            "userID": new_id,
            "username": username,
            "email": email,
            "phone_number": phone_number,
            "password": password,
            "created_at": firestore.SERVER_TIMESTAMP
        })

        return {
            "error": False,
            "message": f"User {username} registered successfully!",
            "userID": str(new_id),
            "email": email,
            "username": username
        }, int(201)

    except Exception as e:
        return {
            "error": True,
            "message": f"An error occurred: {str(e)}"
        }, int(500)

@app.route('/login', methods=['POST'])
def login():
    try:
        # Get the data from the request
        data = request.json
        email = data.get('email')
        password = data.get('password')

        # Validate the inputs
        if not email or not password:
            return jsonify({"error": True, "message": "Both email and password are required."}), 400

        # Check if the email exists
        users_collection = db.collection('users')  # Use collection() instead of collections()
        user_query = users_collection.where('email', '==', email).get()
        if not user_query:
            return jsonify({"error": True, "message": "Invalid email or password."}), 401

        # Verify password
        user_data = user_query[0].to_dict()  # Assume email is unique and fetch the first match
        if user_data['password'] != password:
            return jsonify({"error": True, "message": "Invalid email or password."}), 401

        return jsonify({
            "error": False,
            "message": "Login successful.",
            "userID": str(user_data["userID"]),
            "username": user_data["username"],
            "email": email,
            "phone_number": user_data.get("phone_number", "")
        }), 200
    except Exception as e:
        return jsonify({"error": True, "message": str(e)}), 500

def preprocess_input_text(input_text):
    # Ubah teks menjadi huruf kecil dan hilangkan tanda baca
    input_text = input_text.lower()
    input_text = re.sub(f"[{string.punctuation}]", "", input_text)
    return input_text

# Fungsi untuk memprediksi kategori pekerjaan berdasarkan input pengguna
def predict_job_category(user_input, tfidf_vectorizer, model, label_encoder,job_categories):
    # Praproses teks input pengguna
    preprocessed_text = preprocess_input_text(user_input)

    # Lakukan transformasi TF-IDF pada teks input
    text_vectorized = tfidf_vectorizer.transform([preprocessed_text])  # Input sebagai list

    # Pastikan input menjadi array 2D (shape: 1, n_features)
    text_vectorized_array = text_vectorized.toarray()  # Mengubahnya menjadi array 2D

    # Pad the input tensor to match the expected shape
    padded_input = np.pad(text_vectorized_array, ((0, 0), (0, 437 - text_vectorized_array.shape[1])), 'constant')

    # Convert the input tensor to a float tensor
    input_tensor = tf.convert_to_tensor(padded_input, dtype=tf.float32)

    # Prediksi kategori menggunakan model
    predictions = model.signatures['serving_default'](input_tensor)  # Prediksi hasil
    y_pred = predictions['output_0'].numpy()  # Ambil output dari model
    y_pred_classes = np.argmax(y_pred, axis=1)  # Ambil kelas dengan probabilitas tertinggi

    if y_pred_classes[0] < len(job_categories):
        predicted_category = label_encoder.inverse_transform(y_pred_classes)[0]
    else:
        predicted_category = "Other"

    return predicted_category

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)