# Image Classification for Defective Materials
# Overview of code

---

## Data Loading, Exploration & Preprocessing - (`analyze_sets.py`, `preprocessing.py`)
## Status: Complete

### Objectives
- Load and explore the casting product dataset

### Deliverables
- [x] Working Python scripts in `src/` folder
- [x] Basic statistics about the dataset
- [x] Sample visualizations of images

### Next Steps
Now that preprocessing is complete, the preprocessed data is ready for model training:

**Available Data Files** (in `src/data/` folder):
- `X_train.npy`, `y_train.npy` - Training set (80% of original training data)
- `X_val.npy`, `y_val.npy` - Validation set (20% of original training data)
- `X_test.npy`, `y_test.npy` - Test set (separate from training)
- `metadata.pkl` - Configuration details (image size: 128x128, random seed, etc.)

**Data Specifications**:
- Image size: 128x128 pixels, grayscale
- Shape: (num_samples, 128, 128, 1)
- Pixel values: Normalized to [0, 1] range
- Labels: 0 = OK, 1 = Defective
- Class balance: Maintained through stratified splitting

**To proceed with Model Training**:
1. Load the preprocessed numpy files using `np.load()`
2. Flatten the images from (128, 128, 1) to (16384,) - just a 1D array of pixel values
3. Use simple scikit-learn classifiers:
   - **Logistic Regression** 
   - **Random Forest**
   - **Support Vector Machine (SVM)**
4. Train on training set, validate on validation set
5. Pick the best model and save it for final testing

---

## Model Training & Validation - (`train_model.py`)
## Status: Not Started

### Objectives
- Train a model on the preprocessed data

### Deliverables
- [ ] Working Python script in `src/` folder
- [ ] Trained model
- [ ] Evaluation metrics

---

## Model Evaluation - (`evaluate_model.py`)

### Objectives
- Evaluate the model on the test set

### Deliverables
- [ ] Working Python script in `src/` folder
- [ ] Evaluation metrics

---

## Model Deployment - (`deploy_model.py`)
## Status: Not Started

### Objectives
- Deploy the model for demonstration for project presentation and report

### Deliverables
- [ ] Working Python script in `src/` folder
- [ ] Deployed model
- [ ] Deployment instructions
- [ ] Presentation of the model and relevant results and analysis