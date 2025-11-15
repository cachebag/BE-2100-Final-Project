import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load the preprocessed numpy files from data/ folder
# Flatten the images and return X_train, y_train, X_val, y_val, X_test, y_test
def load_data():

    data_path = os.path.join("data/")

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


# Train 3 models (Logistic Regression, Random Forest, SVM)
# Calculate accuracy for each on training and validation sets
# Return a dictionary: results[model_name] = {'model': model_object, 'train_acc': score, 'val_acc': score}
def train_models(X_train, y_train, X_val, y_val):
    results = {}
    
    # Train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train))
    lr_val_acc = accuracy_score(y_val, lr_model.predict(X_val))
    results['LogisticRegression'] = {
        'model': lr_model,
        'train_acc': lr_train_acc,
        'val_acc': lr_val_acc
    }
    print(f"Logistic Regression - Train Acc: {lr_train_acc:.4f}, Val Acc: {lr_val_acc:.4f}")
    
    # Train Random Forest
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_train_acc = accuracy_score(y_train, rf_model.predict(X_train))
    rf_val_acc = accuracy_score(y_val, rf_model.predict(X_val))
    results['RandomForest'] = {
        'model': rf_model,
        'train_acc': rf_train_acc,
        'val_acc': rf_val_acc
    }
    print(f"Random Forest - Train Acc: {rf_train_acc:.4f}, Val Acc: {rf_val_acc:.4f}")
    
    # Train SVM
    print("\nTraining SVM...")
    svm_model = SVC(random_state=42)
    svm_model.fit(X_train, y_train)
    svm_train_acc = accuracy_score(y_train, svm_model.predict(X_train))
    svm_val_acc = accuracy_score(y_val, svm_model.predict(X_val))
    results['SVM'] = {
        'model': svm_model,
        'train_acc': svm_train_acc,
        'val_acc': svm_val_acc
    }
    print(f"SVM - Train Acc: {svm_train_acc:.4f}, Val Acc: {svm_val_acc:.4f}")
    
    return results

# Find the best model based on validation accuracy
# Access results using: results[model_name]['val_acc']
# Save the best model to models/best_model.pkl
# Return the best model name and model object
def save_best_model(results):
    # Find the best model based on validation accuracy
    best_name = None
    best_val_acc = -1
    best_model = None
    
    for model_name, model_data in results.items():
        val_acc = model_data['val_acc']
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_name = model_name
            best_model = model_data['model']
    
    print(f"\nBest model: {best_name} with validation accuracy: {best_val_acc:.4f}")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join("src", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the best model
    model_path = os.path.join(models_dir, "best_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    # Save training results summary
    training_summary = {
        'best_model_name': best_name,
        'best_val_accuracy': best_val_acc,
        'all_results': {name: {'train_acc': data['train_acc'], 'val_acc': data['val_acc']} 
                       for name, data in results.items()}
    }
    
    summary_path = os.path.join(models_dir, "training_summary.pkl")
    with open(summary_path, 'wb') as f:
        pickle.dump(training_summary, f)
    
    print(f"Best model saved to {model_path}")
    print(f"Training summary saved to {summary_path}")
    
    return best_name, best_model

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    results = train_models(X_train, y_train, X_val, y_val)
    
    best_name, best_model = save_best_model(results)
    
    print("\nDone")

