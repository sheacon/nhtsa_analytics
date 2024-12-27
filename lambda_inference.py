import json
import boto3
import requests
import pandas as pd
import io
import joblib
from sklearn.linear_model import LogisticRegression

# Helper functions to fetch data
def fetch_makes(model_year, issue_type):
    base_url = "https://api.nhtsa.gov/products/vehicle/"
    makes_url = f"{base_url}makes?modelYear={model_year}&issueType={issue_type}"
    response = requests.get(makes_url)
    makes_data = response.json()

    if 'results' not in makes_data:
        raise ValueError("Failed to fetch makes data.")

    return [make['make'] for make in makes_data['results']]

def fetch_models(model_year, makes_list, issue_type):
    base_url = "https://api.nhtsa.gov/products/vehicle/"
    models = []
    for make in makes_list:
        models_url = f"{base_url}models?modelYear={model_year}&make={make}&issueType={issue_type}"
        response = requests.get(models_url)
        models_data = response.json()
        if 'results' in models_data:
            models.extend([
                {
                    'modelYear': model['modelYear'],
                    'make': model['make'],
                    'model': model['model']
                } for model in models_data['results']
            ])
    return pd.DataFrame(models)

def fetch_complaints(models_df):
    complaints = []
    for _, row in models_df.iterrows():
        complaints_url = f"https://api.nhtsa.gov/complaints/complaintsByVehicle?make={row['make']}&model={row['model']}&modelYear={row['modelYear']}"
        response = requests.get(complaints_url)
        complaints_data = response.json()
        if 'results' in complaints_data:
            complaints.extend([
                {
                    'make': row['make'],
                    'model': row['model'],
                    'modelYear': row['modelYear'],
                    'odiNumber': complaint['odiNumber'],
                    'manufacturer': complaint['manufacturer'],
                    'crash': complaint['crash'],
                    'fire': complaint['fire'],
                    'numberOfInjuries': complaint['numberOfInjuries'],
                    'numberOfDeaths': complaint['numberOfDeaths'],
                    'summary': complaint.get('summary')
                } for complaint in complaints_data['results']
            ])
    return pd.DataFrame(complaints)

def lambda_handler(event, context):
    # AWS resource clients
    s3 = boto3.client('s3')

    # Parameters
    bucket_name = "your-s3-bucket-name"
    model_s3_key = "models/model.joblib"
    output_s3_key = "predictions/2021_predictions.csv"
    model_year = 2021

    # Load the model from S3
    model_file = "/tmp/model.joblib"
    s3.download_file(bucket_name, model_s3_key, model_file)
    model = joblib.load(model_file)

    # Fetch data
    makes_list = fetch_makes(model_year, issue_type='c')
    models_df = fetch_models(model_year, makes_list, issue_type='c')
    complaints_df = fetch_complaints(models_df)

    # Prepare data for prediction
    X = complaints_df.drop(columns=['make', 'model', 'modelYear', 'odiNumber', 'manufacturer', 'summary'])

    # Make predictions
    predictions = model.predict(X)
    complaints_df['prediction'] = predictions

    # Save predictions to S3
    csv_buffer = io.StringIO()
    complaints_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=output_s3_key, Body=csv_buffer.getvalue())

    return {
        "statusCode": 200,
        "body": json.dumps(f"Predictions saved to s3://{bucket_name}/{output_s3_key}")
    }
