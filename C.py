import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Id3class import *
import matplotlib.pyplot as plt

df_train = pd.read_csv("adult.train.10k.discrete", header=None)
df_test = pd.read_csv("adult.test.10k.discrete", header=None)
columns = ['target', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
df_train.columns = df_test.columns = columns
df_train['target'] = df_train['target'].replace(['>50K'], 1)
df_train['target'] = df_train['target'].replace(['<=50K'], 0)
df_test['target'] = df_test['target'].replace(['>50K'], 1)
df_test['target'] = df_test['target'].replace(['<=50K'], 0)

X_train = df_train.drop(columns="target")
y_train = df_train["target"]
X1 = df_test.drop(columns="target")
y1 = df_test["target"]
X_test, X_validation, y_test, y_validation = train_test_split(X1, y1, test_size=0.75)

model = Id3class()

model.train(X_train, y_train)

size_before_prunning = model.get_size_of_tree()

y_pred_train = model.predict(X_train, False, 0)
accuracy_train_before_prunning = accuracy_score(y_train, y_pred_train)

y_pred_validation = model.predict(X_validation, False, 0)
accuracy_validation_before_prunning = accuracy_score(y_validation, y_pred_validation)

y_pred_test = model.predict(X_test, False, 0)
accuracy_test_before_prunning = accuracy_score(y_test, y_pred_test)

model.prune(X_validation, y_validation)

accuracy_train_arr, accuracy_validation_arr, accuracy_test_arr, size_arr = [], [], [], []

for i in range(10, int(model.get_size_of_tree()/10), 10):
    size_arr.append(i)
    y_pred_train = model.predict(X_train, True, i)
    accuracy_train_arr.append(accuracy_score(y_train, y_pred_train))

    y_pred_validation = model.predict(X_validation, True, i)
    accuracy_validation_arr.append(accuracy_score(y_validation, y_pred_validation))

    y_pred_test = model.predict(X_test, True, i)
    accuracy_test_arr.append(accuracy_score(y_test, y_pred_test))
    
size_arr.append(model.get_size_of_tree())
y_pred_train = model.predict(X_train, False, 0)
accuracy_train_arr.append(accuracy_score(y_train, y_pred_train))

y_pred_validation = model.predict(X_validation, False, 0)
accuracy_validation_arr.append(accuracy_score(y_validation, y_pred_validation))

y_pred_test = model.predict(X_test, False, 0)
accuracy_test_arr.append(accuracy_score(y_test, y_pred_test))

size_arr.append(size_before_prunning)
accuracy_train_arr.append(accuracy_train_before_prunning)
accuracy_validation_arr.append(accuracy_validation_before_prunning)
accuracy_test_arr.append(accuracy_test_before_prunning)

print('size of tree before prunning : ' + str(size_before_prunning))
print('size of tree after prunning : ' + str(model.get_size_of_tree()))

plt.plot(size_arr, accuracy_train_arr, label='train dataset')
plt.plot(size_arr, accuracy_validation_arr, label='validation dataset')
plt.plot(size_arr, accuracy_test_arr, label='test dataset')

plt.xlabel('size')
plt.ylabel('accuracy')
plt.legend()
plt.show()