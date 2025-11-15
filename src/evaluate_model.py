import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, roc_auc_score
)
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_data():
    """Load all datasets"""
    data_path = os.path.join("data/")
    
    X_train = np.load(os.path.join(data_path, "X_train.npy"))
    y_train = np.load(os.path.join(data_path, "y_train.npy"))
    X_val = np.load(os.path.join(data_path, "X_val.npy"))
    y_val = np.load(os.path.join(data_path, "y_val.npy"))
    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    
    # Flatten images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    print("Data loaded successfully")
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_model():
    """Load the best model and training summary"""
    models_dir = os.path.join("src", "models")
    
    model_path = os.path.join(models_dir, "best_model.pkl")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Try to load training summary
    summary_path = os.path.join(models_dir, "training_summary.pkl")
    training_summary = None
    if os.path.exists(summary_path):
        with open(summary_path, 'rb') as f:
            training_summary = pickle.load(f)
    
    print(f"Model loaded from {model_path}")
    if training_summary:
        print(f"Model type: {training_summary['best_model_name']}")
    
    return model, training_summary

def evaluate_model(model, X, y, set_name="Test"):
    """Evaluate model and return predictions and probabilities"""
    print(f"\nEvaluating on {set_name} set...")
    
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        # For SVM without probability, use decision function
        if hasattr(model, 'decision_function'):
            decision_scores = model.decision_function(X)
            # Normalize to [0, 1] for visualization
            y_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
        else:
            y_proba = None
    
    return {
        'y_true': y,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy,
        'set_name': set_name
    }

def plot_confusion_matrix(y_true, y_pred, set_name, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['OK', 'Defective'],
                yticklabels=['OK', 'Defective'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {set_name} Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def plot_roc_curve(y_true, y_proba, set_name, save_path):
    """Plot and save ROC curve"""
    if y_proba is None:
        print(f"Skipping ROC curve for {set_name} (no probability scores available)")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {set_name} Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {save_path}")
    return roc_auc

def plot_precision_recall_curve(y_true, y_proba, set_name, save_path):
    """Plot and save Precision-Recall curve"""
    if y_proba is None:
        print(f"Skipping PR curve for {set_name} (no probability scores available)")
        return None
    
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {set_name} Set', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Precision-Recall curve to {save_path}")
    return pr_auc

def plot_sample_predictions(X_test, y_test, y_pred, y_proba, save_path, n_samples=16):
    """Plot sample test images with predictions"""
    # Reshape back to image format (assuming 128x128)
    img_size = int(np.sqrt(X_test.shape[1]))
    X_test_images = X_test.reshape(-1, img_size, img_size)
    
    # Find correct and incorrect predictions
    correct = np.where(y_test == y_pred)[0]
    incorrect = np.where(y_test != y_pred)[0]
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    # Show 8 correct and 8 incorrect predictions
    indices = list(correct[:8]) + list(incorrect[:8])
    
    for idx, ax in enumerate(axes):
        if idx < len(indices):
            img_idx = indices[idx]
            img = X_test_images[img_idx]
            true_label = 'OK' if y_test[img_idx] == 0 else 'Defective'
            pred_label = 'OK' if y_pred[img_idx] == 0 else 'Defective'
            prob = y_proba[img_idx] if y_proba is not None else None
            
            ax.imshow(img, cmap='gray')
            color = 'green' if y_test[img_idx] == y_pred[img_idx] else 'red'
            title = f"True: {true_label}\nPred: {pred_label}"
            if prob is not None:
                title += f"\nProb: {prob:.3f}"
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.suptitle('Sample Test Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample predictions to {save_path}")

def plot_model_comparison(training_summary, save_path):
    """Plot comparison of all models"""
    if training_summary is None:
        print("No training summary available, skipping model comparison")
        return
    
    models = list(training_summary['all_results'].keys())
    train_accs = [training_summary['all_results'][m]['train_acc'] for m in models]
    val_accs = [training_summary['all_results'][m]['val_acc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, train_accs, width, label='Training Accuracy', alpha=0.8)
    bars2 = ax.bar(x + width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Comparison: Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved model comparison to {save_path}")

def save_results_to_file(results_dict, save_path):
    """Save evaluation results to JSON file"""
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results_dict)
    
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Saved results to {save_path}")

def main():
    """Main evaluation function"""
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    # Create outputs directory
    outputs_dir = os.path.join("src", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Load data and model
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    model, training_summary = load_model()
    
    # Evaluate on all sets
    train_results = evaluate_model(model, X_train, y_train, "Training")
    val_results = evaluate_model(model, X_val, y_val, "Validation")
    test_results = evaluate_model(model, X_test, y_test, "Test")
    
    # Calculate detailed metrics for test set
    test_cm = confusion_matrix(test_results['y_true'], test_results['y_pred'])
    test_report = classification_report(
        test_results['y_true'], test_results['y_pred'],
        target_names=['OK', 'Defective'],
        output_dict=True
    )
    
    # Calculate ROC AUC and PR AUC if probabilities available
    test_roc_auc = None
    test_pr_auc = None
    if test_results['y_proba'] is not None:
        test_roc_auc = roc_auc_score(test_results['y_true'], test_results['y_proba'])
        precision, recall, _ = precision_recall_curve(test_results['y_true'], test_results['y_proba'])
        test_pr_auc = auc(recall, precision)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"\nTraining Accuracy:   {train_results['accuracy']:.4f}")
    print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
    print(f"Test Accuracy:       {test_results['accuracy']:.4f}")
    
    if training_summary:
        print(f"\nBest Model: {training_summary['best_model_name']}")
        print(f"Best Validation Accuracy: {training_summary['best_val_accuracy']:.4f}")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(test_results['y_true'], test_results['y_pred'],
                              target_names=['OK', 'Defective']))
    
    print(f"\nTest Set Confusion Matrix:")
    print(test_cm)
    print(f"True Negatives: {test_cm[0,0]}, False Positives: {test_cm[0,1]}")
    print(f"False Negatives: {test_cm[1,0]}, True Positives: {test_cm[1,1]}")
    
    if test_roc_auc:
        print(f"\nTest Set ROC AUC: {test_roc_auc:.4f}")
    if test_pr_auc:
        print(f"Test Set PR AUC: {test_pr_auc:.4f}")
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    plot_confusion_matrix(test_results['y_true'], test_results['y_pred'], 
                         "Test", os.path.join(outputs_dir, "confusion_matrix_test.png"))
    
    if test_results['y_proba'] is not None:
        plot_roc_curve(test_results['y_true'], test_results['y_proba'],
                      "Test", os.path.join(outputs_dir, "roc_curve_test.png"))
        plot_precision_recall_curve(test_results['y_true'], test_results['y_proba'],
                                   "Test", os.path.join(outputs_dir, "pr_curve_test.png"))
    
    plot_sample_predictions(X_test, test_results['y_true'], test_results['y_pred'],
                           test_results['y_proba'], 
                           os.path.join(outputs_dir, "sample_predictions.png"))
    
    if training_summary:
        plot_model_comparison(training_summary, 
                             os.path.join(outputs_dir, "model_comparison.png"))
    
    # Compile results dictionary
    results = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_info': {
            'best_model': training_summary['best_model_name'] if training_summary else 'Unknown',
            'best_val_accuracy': training_summary['best_val_accuracy'] if training_summary else None
        },
        'training_metrics': {
            'accuracy': float(train_results['accuracy'])
        },
        'validation_metrics': {
            'accuracy': float(val_results['accuracy'])
        },
        'test_metrics': {
            'accuracy': float(test_results['accuracy']),
            'roc_auc': float(test_roc_auc) if test_roc_auc else None,
            'pr_auc': float(test_pr_auc) if test_pr_auc else None,
            'confusion_matrix': {
                'true_negatives': int(test_cm[0,0]),
                'false_positives': int(test_cm[0,1]),
                'false_negatives': int(test_cm[1,0]),
                'true_positives': int(test_cm[1,1])
            },
            'classification_report': test_report
        },
        'all_model_results': training_summary['all_results'] if training_summary else None
    }
    
    # Save results to JSON
    save_results_to_file(results, os.path.join(outputs_dir, "evaluation_results.json"))
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nAll results and visualizations saved to: {outputs_dir}/")
    print("\nGenerated files:")
    print("  - confusion_matrix_test.png")
    if test_results['y_proba'] is not None:
        print("  - roc_curve_test.png")
        print("  - pr_curve_test.png")
    print("  - sample_predictions.png")
    if training_summary:
        print("  - model_comparison.png")
    print("  - evaluation_results.json")

if __name__ == "__main__":
    main()
