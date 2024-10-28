from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
import joblib

if __name__ == '__main__':

    X_train = pd.read_csv('datasets/train.csv')
    y_train = pd.read_csv('datasets/train_labels.csv')

    rfc = RandomForestClassifier(verbose=1)
    rfc.fit(X_train,y_train)

    joblib.dump(rfc, 'models/rfc.joblib')
