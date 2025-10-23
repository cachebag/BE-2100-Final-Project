import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(data_path):
    dataset = {'train': {'ok': [], 'defective': []}, 'test': {'ok': [], 'defective': []}}
    
    train_ok_dir = os.path.join(data_path, "train", "ok_front")
    train_def_dir = os.path.join(data_path, "train", "def_front")
    test_ok_dir = os.path.join(data_path, "test", "ok_front")
    test_def_dir = os.path.join(data_path, "test", "def_front")
    
    dataset['train']['ok'] = [os.path.join(train_ok_dir, f) for f in os.listdir(train_ok_dir) if f.endswith('.jpeg')]
    dataset['train']['defective'] = [os.path.join(train_def_dir, f) for f in os.listdir(train_def_dir) if f.endswith('.jpeg')]
    dataset['test']['ok'] = [os.path.join(test_ok_dir, f) for f in os.listdir(test_ok_dir) if f.endswith('.jpeg')]
    dataset['test']['defective'] = [os.path.join(test_def_dir, f) for f in os.listdir(test_def_dir) if f.endswith('.jpeg')]
    
    return dataset

def print_stats(dataset):
    train_ok = len(dataset['train']['ok'])
    train_def = len(dataset['train']['defective'])
    test_ok = len(dataset['test']['ok'])
    test_def = len(dataset['test']['defective'])
    
    print("\nDataset Statistics:")
    print(f"Training: {train_ok} OK, {train_def} Defective")
    print(f"Test: {test_ok} OK, {test_def} Defective")
    print(f"Total: {train_ok + train_def + test_ok + test_def} images")

def show_samples(dataset):
    fig, axes = plt.subplots(2, 8, figsize=(15, 4))
    
    for i in range(8):
        img = cv2.imread(dataset['train']['ok'][i], cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
    
    for i in range(8):
        img = cv2.imread(dataset['train']['defective'][i], cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
    
    plt.savefig('outputs/sample_images.png')
    print("Saved sample images")

def plot_distribution(dataset):
    train_ok = len(dataset['train']['ok'])
    train_def = len(dataset['train']['defective'])
    test_ok = len(dataset['test']['ok'])
    test_def = len(dataset['test']['defective'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.bar(['OK', 'Defective'], [train_ok, train_def])
    ax1.set_title('Training Set')
    
    ax2.bar(['OK', 'Defective'], [test_ok, test_def])
    ax2.set_title('Test Set')
    
    plt.savefig('outputs/class_distribution.png')
    print("Saved distribution plot")

if __name__ == "__main__":
    data_path = os.path.expanduser(
        "~/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data"
    )
    
    if not os.path.exists(data_path):
        print("Dataset not found. Run data_set_download.py first.")
        exit()
    
    os.makedirs('outputs', exist_ok=True)
    
    print("Loading data...")
    dataset = load_images(data_path)
    
    print_stats(dataset)
    show_samples(dataset)
    plot_distribution(dataset)