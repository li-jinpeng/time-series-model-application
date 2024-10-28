import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

rfc = joblib.load('models/rfc.joblib')

index2label = {
    0: 'unworn',
    1: 'worn',   
}

# 推理接口
# 输入 shape (batch_size, 47)
# 输出 shape (batch_size)
def inference(inputs):
    results = rfc.predict(inputs)
    labels = []
    for result in results:
        labels.append(index2label[result])
    return results, labels
    
if __name__ == '__main__':
    X_test = pd.read_csv('datasets/test.csv')
    y_test = pd.read_csv('datasets/test_labels.csv')
    results, labels = inference(X_test.values)
    print(results[:10], labels[:10])
    accuracy = accuracy_score(y_test, results)
    print(f'accuracy is {accuracy}')
