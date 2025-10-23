"""
This script loads the dataset, performs data analysis,
and displays key statistics and visualizations.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def load_dataset_paths(data_path):
    """
    Load file paths for all images in the dataset.
    
    Args:
        data_path: Root directory of the dataset
        
    Returns:
        Dictionary containing lists of image paths for each split and class
    """
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


def print_dataset_statistics(dataset):
    """Print basic statistics about the dataset."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Training set
    train_ok_count = len(dataset['train']['ok'])
    train_def_count = len(dataset['train']['defective'])
    train_total = train_ok_count + train_def_count
    
    print(f"\nTraining Set:")
    print(f"  OK Images:        {train_ok_count:,}")
    print(f"  Defective Images: {train_def_count:,}")
    print(f"  Total:            {train_total:,}")
    print(f"  Class Balance:    {train_ok_count/train_total*100:.1f}% OK, "
          f"{train_def_count/train_total*100:.1f}% Defective")
    
    # Test set
    test_ok_count = len(dataset['test']['ok'])
    test_def_count = len(dataset['test']['defective'])
    test_total = test_ok_count + test_def_count
    
    print(f"\nTest Set:")
    print(f"  OK Images:        {test_ok_count:,}")
    print(f"  Defective Images: {test_def_count:,}")
    print(f"  Total:            {test_total:,}")
    print(f"  Class Balance:    {test_ok_count/test_total*100:.1f}% OK, "
          f"{test_def_count/test_total*100:.1f}% Defective")
    
    # Overall
    total_images = train_total + test_total
    print(f"\nOverall Dataset:")
    print(f"  Total Images:     {total_images:,}")
    print(f"  Train/Test Split: {train_total/total_images*100:.1f}% / "
          f"{test_total/total_images*100:.1f}%")
    print("="*60 + "\n")


def analyze_image_properties(dataset, sample_size=100):
    """
    Analyze properties of sample images (dimensions, etc.).
    
    Args:
        dataset: Dictionary of image paths
        sample_size: Number of images to sample from each class
    """
    print("Analyzing image properties...")
    
    dimensions = []
    
    # Sample images from training set
    sample_ok = dataset['train']['ok'][:sample_size]
    sample_def = dataset['train']['defective'][:sample_size]
    
    for img_path in sample_ok + sample_def:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            dimensions.append(img.shape)
    
    # Count unique dimensions
    dim_counter = Counter(dimensions)
    
    print(f"\nImage Dimensions Analysis (sampled {len(dimensions)} images):")
    print(f"  Unique dimensions found: {len(dim_counter)}")
    
    for dim, count in dim_counter.most_common(5):
        print(f"    {dim[0]}x{dim[1]}: {count} images ({count/len(dimensions)*100:.1f}%)")
    
    # Check if all images have same dimensions
    if len(dim_counter) == 1:
        print("  All images have consistent dimensions!")
    else:
        print("  Images have varying dimensions - will need resizing")


def visualize_sample_images(dataset, num_samples=8):
    """
    Display sample images from each class.
    
    Args:
        dataset: Dictionary of image paths
        num_samples: Number of samples to show per class
    """
    print("\nGenerating sample image visualization...")
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 4))
    fig.suptitle('Sample Images: OK (top) vs Defective (bottom)', fontsize=14, fontweight='bold')
    
    # OK samples
    for i in range(num_samples):
        img_path = dataset['train']['ok'][i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('OK', fontweight='bold', color='green')
    
    # Defective samples
    for i in range(num_samples):
        img_path = dataset['train']['defective'][i]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Defective', fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig('outputs/sample_images.png', dpi=150, bbox_inches='tight')
    print("  Saved visualization to 'outputs/sample_images.png'")
    plt.show()


def plot_class_distribution(dataset):
    """Plot bar chart showing class distribution."""
    print("\nGenerating class distribution plot...")
    
    # Calculate counts
    train_ok = len(dataset['train']['ok'])
    train_def = len(dataset['train']['defective'])
    test_ok = len(dataset['test']['ok'])
    test_def = len(dataset['test']['defective'])
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training set
    categories = ['OK', 'Defective']
    counts = [train_ok, train_def]
    colors = ['#2ecc71', '#e74c3c']
    
    ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.1)
    
    # Add count labels on bars
    for i, (cat, count) in enumerate(zip(categories, counts)):
        ax1.text(i, count + max(counts)*0.02, f'{count:,}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Test set
    counts_test = [test_ok, test_def]
    ax2.bar(categories, counts_test, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Number of Images', fontsize=12)
    ax2.set_title('Test Set Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(counts_test) * 1.1)
    
    # Add count labels on bars
    for i, (cat, count) in enumerate(zip(categories, counts_test)):
        ax2.text(i, count + max(counts_test)*0.02, f'{count:,}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/class_distribution.png', dpi=150, bbox_inches='tight')
    print("  Saved plot to 'outputs/class_distribution.png'")
    plt.show()


def main():
    """Main execution function for data analysis."""
    print("\n" + "="*60)
    print("Data Exploration")
    print("="*60)
    
    # Dataset path
    data_path = os.path.expanduser(
        "~/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data"
    )
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print("\nERROR: Dataset not found!")
        print(f"Expected location: {data_path}")
        print("\nPlease run 'data_set_download.py' first to download the dataset.")
        return
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset_paths(data_path)
    print("  Dataset loaded successfully!")
    
    # Print statistics
    print_dataset_statistics(dataset)
    
    # Analyze image properties
    analyze_image_properties(dataset)
    
    # Visualize samples
    visualize_sample_images(dataset, num_samples=8)
    
    # Plot class distribution
    plot_class_distribution(dataset)
    
    print("\n" + "="*60)
    print("="*60)
    print("\nNext Steps:")
    print("  1. Review the generated visualizations in 'outputs/' folder")
    print("  2. Run 'preprocessing.py' to preprocess and split the data")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

