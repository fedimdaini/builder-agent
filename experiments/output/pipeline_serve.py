from flask import Flask, request, jsonify
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np

app = Flask('taxi_trip_duration_prediction')

# Load the model from MLflow
model_uri = os.getenv('MODEL_URI')
model = mlflow.pyfunc.load_model(model_uri)

def single_prediction(features, model):
    # Convert the dictionary of features into a DataFrame
    X = pd.DataFrame([features])
    
    # Make prediction with MLflow model
    prediction = model.predict(X)
    
    return float(prediction)

# Define the predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data with trip features
        trip_features = request.get_json(force=True)

        # Make single prediction with JSON data
        prediction = single_prediction(trip_features, model)
        
        # Return prediction as JSON
        response = {
            'prediction': prediction,
            'model_version': model.metadata.model_version
        }
        return jsonify(response)
    
    except Exception as e:
        # If an error occurs, return the error message
        return jsonify({'error': str(e)}), 500

# Define the health endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
