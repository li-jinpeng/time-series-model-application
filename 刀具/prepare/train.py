import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
import joblib

if __name__ == '__main__':

    X_train = pd.read_csv('datasets/train.csv')
    y_train = pd.read_csv('datasets/train_labels.csv')

    classifier=RFC(n_estimators=100,criterion='entropy',random_state=0)
    classifier.fit(X_train, y_train)
    
    joblib.dump(classifier, 'models/rfc.joblib')