import os
import shutil
import numpy as np
from tqdm import tqdm

# set the random seed number
np.random.seed(47)

# directory path to the data sets
data_path = '/home/jmix/data/bsm/Experiment images'
dest_data_path = '/home/jmix/code/bsm-ds-augment/data'

# file masks
file_mask_wm = 'WM'
file_mask_nm = 'NM'

# function to return the number of files in a directory
def file_count(path, mask='*'):

    files = os.listdir(path)
    count = 0

    if mask == '*':
        count = len([file for file in files if
                    os.path.isfile(os.path.join(path, file))])
    else:
        count = len([file for file in files if
                     file.startswith(mask)])
    
    return count


def copy_files(directories, path):

    wm_file_number = 0
    nm_file_number = 0
    
    os.makedirs(os.path.join(dest_data_path, 'wm'), exist_ok=True)
    os.makedirs(os.path.join(dest_data_path, 'nm'), exist_ok=True)

    print('Copying files to WM and NM directories...')

    for directory in directories:

        if directory != 'Control Experiment': # we don't want to utilize the files from the control experiment directory

            files = [file for file in os.listdir(os.path.join(path, directory))
                    if os.path.isfile(os.path.join(path, directory, file))]
            
            for file in tqdm(files):
                
                if file.startswith('WM'):

                    shutil.copy(os.path.join(path, directory, file), os.path.join(dest_data_path, 'wm', f'wm{wm_file_number}.bmp'))
                    wm_file_number += 1

                if file.startswith('NM'):

                    shutil.copy(os.path.join(path, directory, file), os.path.join(dest_data_path, 'nm', f'nm{nm_file_number}.bmp'))
                    nm_file_number += 1


def train_test_split(directories, data_path):

    print('Moving files to train and test directories...')

    for directory in directories: 

        os.makedirs(os.path.join(data_path, 'test', directory), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'train', directory), exist_ok=True)

        total_files_count = 0

        if os.path.isdir(os.path.join(data_path, directory)):

            total_files_count = file_count(os.path.join(data_path, directory))
            print(f'Number of files in the directory {directory}: {total_files_count}')

            files = [file for file in os.listdir(os.path.join(data_path, directory))
                     if os.path.isfile(os.path.join(data_path, directory, file))]
                    
            train_files = np.random.choice(files, int(len(files) * 0.9), replace=False)
            test_files = [f for f in files if f not in train_files]

            for test_file in tqdm(test_files):

                os.rename(os.path.join(data_path, directory, test_file), os.path.join(data_path, 'test', directory, test_file))

            for train_file in tqdm(train_files):

                os.rename(os.path.join(data_path, directory, train_file), os.path.join(data_path, 'train', directory, train_file))

    os.removedirs(os.path.join(data_path, 'wm'))
    os.removedirs(os.path.join(data_path, 'nm'))


def main():

    # set directories to combine and split WM and NM files
    directories = [directory for directory in os.listdir(data_path)
               if os.path.isdir(os.path.join(data_path, directory))]
    
    copy_files(directories, data_path)
    
    # set directories to split train and test data (90%/10% respectively)
    directories = [directory for directory in os.listdir(dest_data_path)
                if os.path.isdir(os.path.join(dest_data_path, directory))]
    
    train_test_split(directories, dest_data_path)


if __name__ == '__main__':
    main()