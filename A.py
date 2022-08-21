import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Id3class import *

#parameters --------------------------------------------
counter = 3
train_size = 0.30
#-------------------------------------------------------

df_train = pd.read_csv("adult.train.10k.discrete", header=None)
df_test = pd.read_csv("adult.test.10k.discrete", header=None)
columns = ['target', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
df_train.columns = df_test.columns = columns
df_train['target'] = df_train['target'].replace(['>50K'], 1)
df_train['target'] = df_train['target'].replace(['<=50K'], 0)
df_test['target'] = df_test['target'].replace(['>50K'], 1)
df_test['target'] = df_test['target'].replace(['<=50K'], 0)

X1 = df_train.drop(columns="target")
y1 = df_train["target"]
X_test = df_test.drop(columns="target")
y_test = df_test["target"]

sum_accuracy = 0.0
sum_accuracy_train = 0.0
sum_tree_size = 0
for i in range(counter):
  X_train, _, y_train, _ = train_test_split(X1, y1, test_size=(1 - train_size))
  model = Id3class()
  model.train(X_train, y_train)
  y_pred_train = model.predict(X_train, False, 0)
  accuracy_train = accuracy_score(y_train, y_pred_train)
  y_pred = model.predict(X_test, False, 0)
  accuracy = accuracy_score(y_test, y_pred)
  tree_size = model.get_size_of_tree()
  sum_accuracy = sum_accuracy + accuracy
  sum_accuracy_train = sum_accuracy_train + accuracy_train
  sum_tree_size = sum_tree_size + tree_size
  print('counter = ' + str(i) + ', accuracy_test = ' + str(accuracy) + ', accuracy_train = ' + str(accuracy_train)  + ', tree_size = ' + str(tree_size))

avg_accuracy = sum_accuracy / counter
avg_accuracy_train = sum_accuracy_train / counter
avg_tree_size = sum_tree_size / counter
print('average accuracy_test = ' + str(avg_accuracy) + ', average accuracy_train = ' + str(avg_accuracy_train) + ', average tree_size = ' + str(avg_tree_size))