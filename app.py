from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import logging
import joblib
import os
import json

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:9000"}})

# Logging
logging.basicConfig(level=logging.INFO)

# Load model listening
model = tf.keras.models.load_model('listening_classifier.h5')
label_encoder = joblib.load('listening_encoder.pkl')

# Load model structure & written
structure_model = tf.keras.models.load_model('writting_classifier.h5')
structure_encoder = joblib.load('writting_encoder.pkl')

@app.route('/')
def home():
    return "🧠 API is running!"

@app.route('/predict', methods=['POST'])
def predict_weak_part():
    try:
        data = request.get_json()
        short = float(data.get('short_err', 0))
        extended = float(data.get('extended_err', 0))
        talk = float(data.get('talk_err', 0))

        for val in [short, extended, talk]:
            if not 0 <= val <= 1:
                return jsonify({'error': 'Nilai error harus antara 0 dan 1'}), 400

        features = np.array([[short, extended, talk,
                              short - extended,
                              short - talk,
                              extended - talk]])

        probs = model.predict(features)[0]
        pred_index = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_index])[0]

        return jsonify({
            'section': 'Listening',
            'predicted_weak_part': pred_label,
            'confidence': round(float(probs[pred_index]), 4),
            'probabilities': {
                label: round(float(prob), 4)
                for label, prob in zip(label_encoder.classes_, probs)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/structure', methods=['POST'])
def predict_structure_part():
    try:
        data = request.get_json()
        structure = float(data.get('structure_err', 0))
        written = float(data.get('written_err', 0))
        diff = structure - written

        for val in [structure, written]:
            if not 0 <= val <= 1:
                return jsonify({'error': 'Nilai error harus antara 0 dan 1'}), 400

        features = np.array([[structure, written, diff]])

        probs = structure_model.predict(features)[0]
        pred_index = np.argmax(probs)
        pred_label = structure_encoder.inverse_transform([pred_index])[0]

        return jsonify({
            'section': 'Structure & Written Expression',
            'predicted_weak_part': pred_label,
            'confidence': round(float(probs[pred_index]), 4),
            'probabilities': {
                label: round(float(prob), 4)
                for label, prob in zip(structure_encoder.classes_, probs)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/api/listening', methods=['GET'])
def get_listening_questions():
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'data/listening.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            questions = json.load(file)
        return jsonify(questions)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/structure', methods=['GET'])
def get_structure_questions():
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'data/writting.json')
        with open(json_path, 'r', encoding='utf-8') as file:
            questions = json.load(file)

        filtered = [
            q for q in questions
            if q.get('part') in ['Structure', 'Written Expression']
        ]
        return jsonify(filtered)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

