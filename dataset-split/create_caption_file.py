import os
import pandas as pd

data_path = '/home/jmix/code/bsm-ds-augment/data/train'

def make_csv():

    directories = [directory for directory in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, directory))]
    
    wm_files = []
    nm_files = []
    wm_description = []
    nm_description = []
    
    for directory in directories:

        files = [file for file in os.listdir(os.path.join(data_path, directory))
                 if os.path.isfile(os.path.join(data_path, directory, file))]
        
        for file in files:

            if file.startswith('wm') and not file.endswith('.csv'):

                wm_files.append(file)
                wm_description.append('bone surface modification cut marks done with meat (with-meat)')

            elif file.startswith('nm') and not file.endswith('.csv'):

                nm_files.append(file)
                nm_description.append('bone surface modification cut marks done without meat (no-meat)')

    df_wm = pd.DataFrame(
        {
            'file_name': wm_files,
            'description': wm_description
        }
    )

    df_nm = pd.DataFrame(
        {
            'file_name': nm_files,
            'description': nm_description
        }
    )

    df_wm.to_csv(os.path.join(data_path, 'wm', 'metadata.csv'), index=False)
    df_nm.to_csv(os.path.join(data_path, 'nm', 'metadata.csv'), index=False)

def main():

    make_csv()

if __name__ == '__main__':
    
    main()