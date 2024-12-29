#!/bin/bash

# Variables
S3_BUCKET="nhtsa-analytics"
S3_STACK_NAME="nhtsa-s3"
S3_TEMPLATE_FILE="nhtsa_s3.yaml"

TRAIN_SCRIPT="lambda_train.py"
TRAIN_PACKAGE="lambda_train_package"
TRAIN_ZIP="lambda_train.zip"

INFERENCE_SCRIPT="lambda_inference.py"
INFERNECE_PACKAGE="lambda_inference_package"
INFERENCE_ZIP="lambda_inference.zip"

PIPELINE_STACK_NAME="nhtsa-pipeline"
PIPELINE_TEMPLATE_FILE="nhtsa_pipeline.yaml"

# Deploy the CloudFormation stack
aws cloudformation deploy \
    --stack-name "$S3_STACK_NAME" \
    --template-file "$S3_TEMPLATE_FILE" \
    --capabilities CAPABILITY_NAMED_IAM

# Output success message
echo "CloudFormation stack '$S3_STACK_NAME' deployed successfully."

# Zip and Upload the Lambda code
mkdir "$TRAIN_PACKAGE"
cd "$TRAIN_PACKAGE"
pip install requests -t .
zip -r "$TRAIN_ZIP" .
aws s3 cp "$TRAIN_ZIP" "s3://$S3_BUCKET/lambda/$TRAIN_ZIP"
cd ..
rmdir -r "$TRAIN_PACKAGE"

mkdir "$INFERENCE_PACKAGE"
cd "$INFERENCE_PACKAGE"
pip install requests -t .
zip -r "$INFERENCE_ZIP" .
aws s3 cp "$INFERENCE_ZIP" "s3://$S3_BUCKET/lambda/$INFERENCE_ZIP"
cd ..
rmdir -r "$INFERENCE_PACKAGE"

# Output success message
echo "Lambda code uploaded successfully."

# Deploy the CloudFormation stack
aws cloudformation deploy \
    --stack-name "$PIPELINE_STACK_NAME" \
    --template-file "$PIPELINE_TEMPLATE_FILE" \
    --capabilities CAPABILITY_NAMED_IAM

# Output success message
echo "CloudFormation stack '$PIPELINE_STACK_NAME' deployed successfully."

# Trigger training lambda
aws lambda invoke --function-name TrainLambda \
    --payload '{"bucket_name": "nhtsa-analytics", "model_year": 2020' \
    response.json
