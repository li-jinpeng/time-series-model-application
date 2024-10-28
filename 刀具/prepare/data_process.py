import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    frames = list()
    results = pd.read_csv("raw_datasets/train.csv")
    for i in range(1,19):
        exp = '0' + str(i) if i < 10 else str(i)
        frame = pd.read_csv("raw_datasets/experiment_{}.csv".format(exp))
        row = results[results['No'] == i]
        frame['target'] = 1 if row.iloc[0]['tool_condition'] == 'worn' else 0
        frames.append(frame)
    df = pd.concat(frames, ignore_index = True)

    x=df.drop(columns=['target','Machining_Process'],axis=1)
    y=df['target']
    

    X_train,X_test,y_train,y_test =train_test_split(x,y,train_size=0.8,random_state=100)
    
    
    X_train.to_csv('datasets/train.csv', index=False)
    X_test.to_csv('datasets/test.csv', index=False)
    y_train.to_csv('datasets/train_labels.csv', index=False)
    y_test.to_csv('datasets/test_labels.csv', index=False)
