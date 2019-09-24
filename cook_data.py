import os
import Cooking


# chunk size for training batches
chunk_size = 32

# No test set needed, since testing in our case is running the model on an unseen map in AirSim
train_val_test_ratio = [0.8, 0.2, 0.0]

# Point this to the directory containing the raw data
RAW_DATA_DIR = './raw_data/'

# Point this to the desired output directory for the cooked (.h5) data
OUTPUT_DIR = './cooked_data/'

# Choose The folders to search for data under RAW_DATA_DIR
COOK_ALL_DATA = True

data_folders = []

# if COOK_ALL_DATA is set to False, append your desired data folders here
# data_folder.append('folder_name1')
# data_folder.append('folder_name2')
# ...
if COOK_ALL_DATA:
    data_folders = [name for name in os.listdir(RAW_DATA_DIR)]


full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in data_folders]
Cooking.cook(full_path_raw_folders, OUTPUT_DIR,
             train_val_test_ratio, chunk_size)
