import kagglehub

# This script downloads the dataset to your computer and will print the path in case you lose it 
# Run with `python data_set_download.py`


path = kagglehub.dataset_download("ravirajsinh45/real-life-industrial-dataset-of-casting-product")

print("Path to dataset files:", path)
