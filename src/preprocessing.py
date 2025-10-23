import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

IMG_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

def load_and_preprocess(paths):
    images = []
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32') / 255.0
        images.append(img)
    return np.array(images)

def get_paths(data_path):
    train_ok_dir = os.path.join(data_path, "train", "ok_front")
    train_def_dir = os.path.join(data_path, "train", "def_front")
    test_ok_dir = os.path.join(data_path, "test", "ok_front")
    test_def_dir = os.path.join(data_path, "test", "def_front")
    
    train_ok = [os.path.join(train_ok_dir, f) for f in os.listdir(train_ok_dir) if f.endswith('.jpeg')]
    train_def = [os.path.join(train_def_dir, f) for f in os.listdir(train_def_dir) if f.endswith('.jpeg')]
    test_ok = [os.path.join(test_ok_dir, f) for f in os.listdir(test_ok_dir) if f.endswith('.jpeg')]
    test_def = [os.path.join(test_def_dir, f) for f in os.listdir(test_def_dir) if f.endswith('.jpeg')]
    
    return train_ok, train_def, test_ok, test_def

if __name__ == "__main__":
    data_path = os.path.expanduser(
        "~/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data"
    )
    
    if not os.path.exists(data_path):
        print("Dataset not found")
        exit()
    
    print("Loading paths...")
    train_ok, train_def, test_ok, test_def = get_paths(data_path)
    
    all_train_paths = train_ok + train_def
    all_train_labels = [0] * len(train_ok) + [1] * len(train_def)
    
    print(f"Total training: {len(all_train_paths)}")
    print("Splitting data...")
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_train_paths, all_train_labels, 
        test_size=VALIDATION_SPLIT, 
        random_state=RANDOM_SEED,
        stratify=all_train_labels
    )
    
    test_paths = test_ok + test_def
    test_labels = [0] * len(test_ok) + [1] * len(test_def)
    
    print("Processing images...")
    X_train = load_and_preprocess(train_paths)
    X_val = load_and_preprocess(val_paths)
    X_test = load_and_preprocess(test_paths)
    
    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)
    
    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_val = X_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    os.makedirs('data', exist_ok=True)
    print("Saving data...")
    
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    metadata = {
        'img_size': IMG_SIZE,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    }
    
    with open('data/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print("Done")