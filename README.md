# BE-2100-Final-Project

## How to Run

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Download dataset:
```bash
python data_set_download.py
```

3. Explore the data:
```bash
cd src
python analyze_sets.py
```
This will create the `outputs/` folder with the following files:
- class_distribution.png - Distribution of the dataset
- sample_images.png - Sample images from the dataset (top row: OK, bottom row: Defective)

4. Preprocess the data:
```bash
python preprocessing.py
```

This creates the train/val/test splits in `data/` folder (root directory). These are going to be used for model training.
- `X_train.npy` - Training set (80% of original training data)
- `y_train.npy` - Training labels (80% of original training data)
- `X_val.npy` - Validation set (20% of original training data)
- `y_val.npy` - Validation labels (20% of original training data)
- `X_test.npy` - Test set (separate from training)
- `y_test.npy` - Test labels (separate from training)
- `metadata.pkl` - Configuration details (image size: 128x128, random seed, etc.)

5. Train the models:
```bash
python src/train_model.py
```

This will:
- Train 3 models (Logistic Regression, Random Forest, SVM)
- Compare their validation accuracies
- Save the best model to `src/models/best_model.pkl`
- Save training summary to `src/models/training_summary.pkl`

6. Evaluate the model:
```bash
python src/evaluate_model.py
```

This will generate comprehensive evaluation results in `src/outputs/`:
- `evaluation_results.json` - Complete numerical results
- `confusion_matrix_test.png` - Confusion matrix visualization
- `roc_curve_test.png` - ROC curve (if available)
- `pr_curve_test.png` - Precision-Recall curve (if available)
- `sample_predictions.png` - Sample test images with predictions
- `model_comparison.png` - Comparison of all trained models

See `src/RESULTS_APPENDIX.md` for detailed explanation of results and metrics.