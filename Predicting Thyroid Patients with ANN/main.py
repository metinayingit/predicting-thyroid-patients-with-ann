import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

train_file_path = 'C:/Users/metinayin/Desktop/ann-train.data'
test_file_path = 'C:/Users/metinayin/Desktop/ann-test.data'

train_data = []
with open(train_file_path, 'r') as file:
    for line in file:
        train_data.append(list(map(float, line.strip().split())))

test_data = []
with open(test_file_path, 'r') as file:
    for line in file:
        test_data.append(list(map(float, line.strip().split())))

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter=500)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))