import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Parse arguments for SageMaker training environment
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    return parser.parse_args()

def main():
    args = parse_args()

    # Load data
    train_data_path = os.path.join(args.train_data, 'training_data.csv')
    data = pd.read_csv(train_data_path)

    # Split into features and labels
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")

    # Save model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)

if __name__ == '__main__':
    main()
