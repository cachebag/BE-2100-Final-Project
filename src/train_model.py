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
    pass

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

