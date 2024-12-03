from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained models
try:
    with open('heating_model.pkl', 'rb') as f:
        heating_model = pickle.load(f)
    with open('cooling_model.pkl', 'rb') as f:
        cooling_model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the frontend
    data = request.json
    print(f"Received data: {data}")  # Debugging output

    if 'features' not in data or 'type' not in data:
        return jsonify({'error': 'Missing features or type parameter'}), 400

    features = np.array([data['features']])

    # Check the features received
    print(f"Features: {features}")  # Debugging output

    # Prediction based on the requested type
    try:
        if data['type'] == 'heating':
            prediction = heating_model.predict(features)
        elif data['type'] == 'cooling':
            prediction = cooling_model.predict(features)
        else:
            return jsonify({'error': 'Invalid prediction type'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    print(f"Prediction: {prediction[0]}")  # Debugging output
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
