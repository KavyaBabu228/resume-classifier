from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Load the vectorizer and model
vectorizer_path = "tfidf.pkl"
model_path = "gradient_boosting.pkl"

if not os.path.exists(vectorizer_path) or not os.path.exists(model_path):
    raise FileNotFoundError("Model or vectorizer file not found. Ensure they are in the correct location.")

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Resume Classification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting {"resume": "your resume text"}
    
    if 'resume' not in data:
        return jsonify({"error": "Missing 'resume' key in request JSON"}), 400

    # Transform input text using the vectorizer
    resume_text = [data['resume']]
    transformed_text = vectorizer.transform(resume_text)

    # Make prediction
    prediction = model.predict(transformed_text)

    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
