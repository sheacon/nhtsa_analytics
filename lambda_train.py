import json
import boto3
import requests
import pandas as pd
import io
from datetime import datetime

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
    sagemaker = boto3.client('sagemaker')

    # Extract parameters from the event payload
    bucket_name = event.get('bucket_name', 'nhtsa-analytics')
    model_year = event.get('model_year', 2020)
    #sagemaker_role = event.get('sagemaker_role', "arn:aws:iam::your-account-id:role/service-role/AmazonSageMaker-ExecutionRole")
    train_runtime = event.get('train_runtime', 600)
    train_instance = event.get('train_instance', "ml.m5.large")
    output_s3_path = f"s3://{bucket_name}/models/"

    # Fetch data
    makes_list = fetch_makes(model_year, issue_type='c')
    models_df = fetch_models(model_year, makes_list, issue_type='c')
    complaints_df = fetch_complaints(models_df)

    # Save data to S3
    data_key = f"data/complaints_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    csv_buffer = io.StringIO()
    complaints_df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket_name, Key=data_key, Body=csv_buffer.getvalue())

    # Launch SageMaker training job
    training_job_name = f"LogisticRegressionJob-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    response = sagemaker.create_training_job(
        TrainingJobName=training_job_name,
        AlgorithmSpecification={
            "TrainingImage": "683313688378.dkr.ecr.us-west-2.amazonaws.com/sagemaker-sklearn:0.23-1-cpu-py3",
            "TrainingInputMode": "File"
        },
        #RoleArn=sagemaker_role,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket_name}/{data_key}",
                        "S3DataDistributionType": "FullyReplicated"
                    }
                },
                "ContentType": "text/csv"
            }
        ],
        OutputDataConfig={
            "S3OutputPath": output_s3_path
        },
        ResourceConfig={
            "InstanceType": train_instance,
            "InstanceCount": 1,
            "VolumeSizeInGB": 10
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": train_runtime
        }
    )

    return {
        "statusCode": 200,
        "body": json.dumps(f"Triggered SageMaker job: {response['TrainingJobArn']}")
    }