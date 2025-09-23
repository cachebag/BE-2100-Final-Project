import os
import cv2
import glob

def main():
    # This is the path to the dataset after downloading with data_set_download.py
    data_path = os.path.expanduser(
        "~/.cache/kagglehub/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product/versions/2/casting_data/casting_data"
    )

    # Make sure the dataset actually exists
    if not os.path.exists(data_path):
        print("Dataset not found. Run data_set_download.py first.")
        return

    # Collect all image file paths for each class and split

    # Training set
    ok_train = glob.glob(os.path.join(data_path, "train", "ok_front", "*.jpeg"))
    def_train = glob.glob(os.path.join(data_path, "train", "def_front", "*.jpeg"))

    # Test set 
    ok_test = glob.glob(os.path.join(data_path, "test", "ok_front", "*.jpeg"))
    def_test = glob.glob(os.path.join(data_path, "test", "def_front", "*.jpeg"))


    # Print counts so we know how many images are in each group
    print(f"Found {len(ok_train)} OK training images")
    print(f"Found {len(def_train)} Defective training images")
    print(f"Found {len(ok_test)} OK test images")
    print(f"Found {len(def_test)} Defective test images")

    # Load one sample image to confirm shape (height x width)
    if ok_train:
        img = cv2.imread(ok_train[0], cv2.IMREAD_GRAYSCALE)
        print(f"Sample image shape: {img.shape}")

# Standard Python entry point: Basically just means run main() when this file is executed
if __name__ == "__main__":
    main()
