#!/usr/bin/env python3
"""
Simple Flask API for HSI Forecast.
Run: python3 api.py
Endpoint: GET http://localhost:5010/predict
"""

from flask import Flask, jsonify
import sys
sys.path.insert(0, '/root/clawd/projects/hsi-forecast/src')
from predict import predict_today

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        result = predict_today()
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5012, debug=False)
