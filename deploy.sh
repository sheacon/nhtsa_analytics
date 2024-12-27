#!/bin/bash

# Variables
S3_BUCKET="your-s3-bucket-name"
STACK_NAME="logistic-pipeline-stack"
TEMPLATE_FILE="logistic_pipeline_infra.yaml"
FETCH_TRAIN_LAMBDA="lambda_fetch_train.py"
PREDICT_LAMBDA="lambda_predict_s3.py"
FETCH_TRAIN_ZIP="fetch_train.zip"
PREDICT_ZIP="predict.zip"

# Step 1: Zip the Lambda code
zip -j "$FETCH_TRAIN_ZIP" "$FETCH_TRAIN_LAMBDA"
zip -j "$PREDICT_ZIP" "$PREDICT_LAMBDA"

# Step 2: Upload Lambda packages to S3
aws s3 cp "$FETCH_TRAIN_ZIP" "s3://$S3_BUCKET/lambda/$FETCH_TRAIN_ZIP"
aws s3 cp "$PREDICT_ZIP" "s3://$S3_BUCKET/lambda/$PREDICT_ZIP"

# Step 3: Deploy the CloudFormation stack
aws cloudformation deploy \
    --stack-name "$STACK_NAME" \
    --template-file "$TEMPLATE_FILE" \
    --capabilities CAPABILITY_NAMED_IAM

# Step 4: Clean up local zip files
rm "$FETCH_TRAIN_ZIP" "$PREDICT_ZIP"

# Output success message
echo "CloudFormation stack '$STACK_NAME' deployed successfully."
