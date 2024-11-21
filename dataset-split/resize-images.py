import os
import shutil
import numpy as np
from tqdm import tqdm
import cv2

data_path = # enter path to training data folder 
dest_data_path = # enter path to training data destination folder
mask = '.bmp'

directories = [directory for directory in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, directory))]

for directory in directories:

    os.makedirs(os.path.join(dest_data_path, directory), exist_ok=True)

    files = [file for file in os.listdir(os.path.join(data_path, directory)) if os.path.isfile(os.path.join(data_path, directory, file))]
    print(f'Number of files in the directory {directory}: {len(files)}')

    for file in tqdm(files):
        
        if file.endswith(mask):
            img = cv2.imread(os.path.join(data_path, directory, file))
            img = cv2.resize(img, (768, 768))
            cv2.imwrite(os.path.join(dest_data_path, directory, file), img)
        else:
            shutil.copy(os.path.join(data_path, directory, file), os.path.join(dest_data_path, directory, file))

print('Done resizing images.')
