import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ASSIGNED TO: Ali
# Load the preprocessed numpy files from data/ folder
# Flatten the images and return X_train, y_train, X_val, y_val, X_test, y_test
def load_data():

    data_path = os.path.join("src", "data")

    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    X_val = np.load(os.path.join(data_path, "X_val.npy"))
    y_val = np.load(os.path.join(data_path, "y_val.npy"))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))

    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    print("Data loaded and flattened successfully")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ASSIGNED TO: Abdoul
# Train 3 models (Logistic Regression, Random Forest, SVM)
# Calculate accuracy for each on training and validation sets
# Return a dictionary: results[model_name] = {'model': model_object, 'train_acc': score, 'val_acc': score}
def train_models(X_train, y_train, X_val, y_val):
    pass

# ASSIGNED TO: Caitlin
# Find the best model based on validation accuracy
# Access results using: results[model_name]['val_acc']
# Save the best model to models/best_model.pkl
# Return the best model name and model object
def save_best_model(results):
    pass

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    results = train_models(X_train, y_train, X_val, y_val)
    
    best_name, best_model = save_best_model(results)
    
    print("\nDone")

