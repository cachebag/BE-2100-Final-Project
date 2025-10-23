"""
Data Preprocessing and Splitting
Defective Casting Product Classification

This script preprocesses images and splits the training data
into training and validation sets.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle


# Configuration
IMG_SIZE = 128  # Resize images to 128x128 (good balance for simple models)
VALIDATION_SPLIT = 0.2  # 20% of training data for validation
RANDOM_SEED = 42  # For reproducibility


def load_and_preprocess_images(image_paths, img_size=IMG_SIZE):
    """
    Load images from paths and preprocess them.
    
    Args:
        image_paths: List of image file paths
        img_size: Target size for resizing (will be img_size x img_size)
        
    Returns:
        Numpy array of preprocessed images
    """
    images = []
    
    for img_path in image_paths:
        # Read image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Resize to target size
        img = cv2.resize(img, (img_size, img_size))
        
        # Normalize pixel values to [0, 1]
        img = img.astype('float32') / 255.0
        
        images.append(img)
    
    return np.array(images)


def load_dataset_paths(data_path):
    """Load file paths for all images in the dataset."""
    dataset = {
        'train': {
            'ok': [],
            'defective': []
        },
        'test': {
            'ok': [],
            'defective': []
        }
    }
    
    # Training set
    train_ok_dir = os.path.join(data_path, "train", "ok_front")
    train_def_dir = os.path.join(data_path, "train", "def_front")
    
    if os.path.exists(train_ok_dir):
        dataset['train']['ok'] = [
            os.path.join(train_ok_dir, f) 
            for f in os.listdir(train_ok_dir) 
            if f.endswith('.jpeg')
        ]
    
    if os.path.exists(train_def_dir):
        dataset['train']['defective'] = [
            os.path.join(train_def_dir, f) 
            for f in os.listdir(train_def_dir) 
            if f.endswith('.jpeg')
        ]
    
    # Test set
    test_ok_dir = os.path.join(data_path, "test", "ok_front")
    test_def_dir = os.path.join(data_path, "test", "def_front")
    
    if os.path.exists(test_ok_dir):
        dataset['test']['ok'] = [
            os.path.join(test_ok_dir, f) 
            for f in os.listdir(test_ok_dir) 
            if f.endswith('.jpeg')
        ]
    
    if os.path.exists(test_def_dir):
        dataset['test']['defective'] = [
            os.path.join(test_def_dir, f) 
            for f in os.listdir(test_def_dir) 
            if f.endswith('.jpeg')
        ]
    
    return dataset


def prepare_data(dataset):
    """
    Prepare and split the dataset.
    
    Returns:
        Dictionary containing train, validation, and test sets
    """
    print("\n" + "="*60)
    print("PREPARING DATASET")
    print("="*60)
    
    # Combine paths and create labels for training data
    train_ok_paths = dataset['train']['ok']
    train_def_paths = dataset['train']['defective']
    
    all_train_paths = train_ok_paths + train_def_paths
    all_train_labels = [0] * len(train_ok_paths) + [1] * len(train_def_paths)  # 0=OK, 1=Defective
    
    print(f"\nTotal training samples: {len(all_train_paths)}")
    print(f"  - OK: {len(train_ok_paths)}")
    print(f"  - Defective: {len(train_def_paths)}")
    
    # Split training data into train and validation
    print(f"\nSplitting into train ({int((1-VALIDATION_SPLIT)*100)}%) "
          f"and validation ({int(VALIDATION_SPLIT*100)}%) sets...")
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths,
        all_train_labels,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=all_train_labels  # Maintain class balance
    )
    
    print(f"  ✓ Training samples: {len(train_paths)}")
    print(f"  ✓ Validation samples: {len(val_paths)}")
    
    # Prepare test data
    test_ok_paths = dataset['test']['ok']
    test_def_paths = dataset['test']['defective']
    
    test_paths = test_ok_paths + test_def_paths
    test_labels = [0] * len(test_ok_paths) + [1] * len(test_def_paths)
    
    print(f"\nTest samples: {len(test_paths)}")
    print(f"  - OK: {len(test_ok_paths)}")
    print(f"  - Defective: {len(test_def_paths)}")
    
    # Load and preprocess images
    print(f"\nLoading and preprocessing images (resizing to {IMG_SIZE}x{IMG_SIZE})...")
    
    print("  Processing training set...")
    X_train = load_and_preprocess_images(train_paths)
    y_train = np.array(train_labels)
    
    print("  Processing validation set...")
    X_val = load_and_preprocess_images(val_paths)
    y_val = np.array(val_labels)
    
    print("  Processing test set...")
    X_test = load_and_preprocess_images(test_paths)
    y_test = np.array(test_labels)
    
    # Reshape for CNN (add channel dimension)
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    print(f"\n  Training data shape: {X_train.shape}")
    print(f"    Validation data shape: {X_val.shape}")
    print(f"    Test data shape: {X_test.shape}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def save_preprocessed_data(data, output_dir='data'):
    """Save preprocessed data to disk."""
    print(f"\nSaving preprocessed data to '{output_dir}/'...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy files (efficient for numerical data)
    np.save(os.path.join(output_dir, 'X_train.npy'), data['X_train'])
    np.save(os.path.join(output_dir, 'y_train.npy'), data['y_train'])
    np.save(os.path.join(output_dir, 'X_val.npy'), data['X_val'])
    np.save(os.path.join(output_dir, 'y_val.npy'), data['y_val'])
    np.save(os.path.join(output_dir, 'X_test.npy'), data['X_test'])
    np.save(os.path.join(output_dir, 'y_test.npy'), data['y_test'])
    
    # Save metadata
    metadata = {
        'img_size': IMG_SIZE,
        'train_samples': len(data['X_train']),
        'val_samples': len(data['X_val']),
        'test_samples': len(data['X_test']),
        'validation_split': VALIDATION_SPLIT,
        'random_seed': RANDOM_SEED
    }
    
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)
    
    print("  All data saved successfully!")
    print("\nSaved files:")
    print(f"    X_train.npy, y_train.npy")
    print(f"    X_val.npy, y_val.npy")
    print(f"    X_test.npy, y_test.npy")
    print(f"    metadata.pkl")


def print_summary(data):
    """Print summary statistics of the preprocessed data."""
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY")
    print("="*60)
    
    print(f"\nImage size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"Pixel value range: [0, 1] (normalized)")
    print(f"Data type: {data['X_train'].dtype}")
    
    print(f"\nTraining Set:")
    print(f"  Shape: {data['X_train'].shape}")
    print(f"  OK samples: {np.sum(data['y_train'] == 0)} ({np.sum(data['y_train'] == 0) / len(data['y_train']) * 100:.1f}%)")
    print(f"  Defective samples: {np.sum(data['y_train'] == 1)} ({np.sum(data['y_train'] == 1) / len(data['y_train']) * 100:.1f}%)")
    
    print(f"\nValidation Set:")
    print(f"  Shape: {data['X_val'].shape}")
    print(f"  OK samples: {np.sum(data['y_val'] == 0)} ({np.sum(data['y_val'] == 0) / len(data['y_val']) * 100:.1f}%)")
    print(f"  Defective samples: {np.sum(data['y_val'] == 1)} ({np.sum(data['y_val'] == 1) / len(data['y_val']) * 100:.1f}%)")
    
    print(f"\nTest Set:")
    print(f"  Shape: {data['X_test'].shape}")
    print(f"  OK samples: {np.sum(data['y_test'] == 0)} ({np.sum(data['y_test'] == 0) / len(data['y_test']) * 100:.1f}%)")
    print(f"  Defective samples: {np.sum(data['y_test'] == 1)} ({np.sum(data['y_test'] == 1) / len(data['y_test']) * 100:.1f}%)")
    
    print("="*60 + "\n")


def main():
    """Main execution function for data preprocessing."""
    print("\n" + "="*60)
    print("Data Preprocessing")
    print("="*60)
    
    data_path = os.path.expanduser(
        "~/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data"
    )
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print("\nERROR: Dataset not found!")
        print(f"Expected location: {data_path}")
        print("\nPlease run 'data_set_download.py' first to download the dataset.")
        return
    
    print("\nLoading dataset structure...")
    dataset = load_dataset_paths(data_path)
    print("  Dataset structure loaded!")
    
    data = prepare_data(dataset)
    
    save_preprocessed_data(data)
    
    print_summary(data)
    
    print("="*60)
    print("="*60)
    print("\nData is ready for model training!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

