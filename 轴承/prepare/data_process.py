import numpy as np
import pandas as pd 
import glob
from sklearn.model_selection import train_test_split

root_path = 'raw_datasets'

normal_file_names = glob.glob(f"{root_path}/normal/*.csv")

def dataReader(path_names):
    data_full = pd.DataFrame()
    
    for i in path_names:
        low_data = pd.read_csv(i,header=None)
        data_full = pd.concat([data_full,low_data],ignore_index=True)
    
    data_full.rename(mapper={ 0: 'RF',
                    1: 'underhang bearing ax',
                    2: 'underhang bearing rad',
                    3: 'underhang bearing tan',
                    4: 'overhang bearing ax',
                    5: 'overhang bearing rad',
                    6: 'overhang bearing tan',
                    7: 'microphone' }, axis=1, inplace=True)
        
    return data_full

def downSampler(data, a, b):
    """
    data = data
    a = start index
    b = sampling rate
    """

    data_decreased = pd.DataFrame()

    x = b

    for i in range(int(len(data)/x)):
        data_mean = data.iloc[a:b,:].sum() / x
        data_decreased = pd.concat([data_decreased, data_mean.to_frame().T])
        a += x
        b += x

    return data_decreased


if __name__ == '__main__':
    data_type = 'horizontal-misalignment'
    
    imnormal_file_names_05mm = glob.glob(f'{root_path}/{data_type}/0.5mm/*.csv')
    imnormal_file_names_10mm = glob.glob(f'{root_path}/{data_type}/1.0mm/*.csv')
    imnormal_file_names_15mm = glob.glob(f'{root_path}/{data_type}/1.5mm/*.csv')
    imnormal_file_names_20mm = glob.glob(f'{root_path}/{data_type}/2.0mm/*.csv')
    
    data_n = dataReader(normal_file_names)

    data_05mm = dataReader(imnormal_file_names_05mm)
    data_10mm = dataReader(imnormal_file_names_10mm)
    data_15mm = dataReader(imnormal_file_names_15mm)
    data_20mm = dataReader(imnormal_file_names_20mm)
    
    data_n = downSampler(data_n, 0,2000)

    data_05mm = downSampler(data_05mm, 0,2000)
    data_10mm = downSampler(data_10mm, 0,2000)
    data_15mm = downSampler(data_15mm, 0,2000)
    data_20mm = downSampler(data_20mm, 0,2000)                                                                                  

    y_n = pd.DataFrame(np.zeros(int(len(data_n)),dtype=int))

    y_05mm = pd.DataFrame(np.ones(int(len(data_05mm)),dtype=int))
    y_10mm = pd.DataFrame(np.full((int(len(data_10mm)),1),1))
    y_15mm = pd.DataFrame(np.full((int(len(data_15mm)),1),1))
    y_20mm = pd.DataFrame(np.full((int(len(data_20mm)),1),1))


    y = pd.concat([y_n,y_05mm,y_10mm,y_15mm,y_20mm], ignore_index=True)
    data = pd.concat([data_n,data_05mm ,data_10mm,data_15mm ,data_20mm],ignore_index=True)
    
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.20, shuffle=True)
    
    X_train.to_csv('datasets/train.csv',  index=False)
    X_test.to_csv('datasets/test.csv', index=False)
    y_train.to_csv('datasets/train_labels.csv', index=False)
    y_test.to_csv('datasets/test_labels.csv', index=False)
        