# Import the necessary libraries
import boto3
import json
from sklearn.model_selection import HalvingGridSearchCV
from flask import Flask, request, jsonify
import pandas as pd

# Instantiate Flask app
app = Flask(__name__)

# Define the model path
# When you configure the model, you will need to specify the S3 location of your model artifacts.
# Sagemaker will automatically download, decompress and store the model's weights in the /opt/ml/model folder.
MODEL_PATH = "/opt/ml/model/s3://sagemaker-us-east-2-482523031755/sagemaker-scikit-learn-2023-01-02-00-16-10-673/output/model.tar.gz"

# Load the CatBoost model from the specified path
model = HalvingGridSearchCV().load_model(MODEL_PATH)

# Define an endpoint for health check
@app.route('/ping', methods=['GET'])
def ping():
  return '', 200

# Define an endpoint for making predictions
@app.route('/invocations', methods=['POST'])
def predict():
  # Get data from the POST request
  data = request.get_data().decode('utf-8')

  # Convert the data into a Pandas DataFrame
  df = pd.read_json(data, orient='split')
  
  # Make predictions using the loaded model
  prediction = model.predict(df)

  # Return the prediction results as JSON
  return json.dumps(prediction.tolist())