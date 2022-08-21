# Implementation of decision tree algorithm for dataset adult

https://archive.ics.uci.edu/ml/datasets/adult 
__________________________________________________________
The data given under the name Adult is extracted from the UCI collection. Its purpose is a binary classifier to predict whether a person earns more than 50K per year or not. In its initial form, this data contains 14 features, 8 of which are discrete categorical and 6 of which are continuous, and in addition, it has a class label feature. For simplification and the possibility of implementing ID3, we have removed continuous features from the data set. Also, a number of examples included features with unknown value (missing), which were also removed. As a result of the existing data, the clean data is of 8 categorical features, which is divided into 2 sets of train and test, each of which has 10,000 examples. The first feature of each example specifies the data label and the other 8 features contain the following values:

1) workclass (8 values)

2) education (16 values)

3) marital-status (7 values)

4) occupation (14 values)

5) relationship (6 values)

6) race (5 values)

7) sex (2 values)

8) native-country (41 values)
_____________________________________________________________
We make a decision tree with the ID3 algorithm and the train data set. To select the best feature in each step, we must use information gain.

A) In part A, we randomly select the percentage of training data. And we train the tree with it. We repeat this process of random data division for the training process 3 times.

B) Measuring the effect of the size of the training data: In addition to the random selection of 30% of the training data that we did in the previous step, we also randomly select the values of 40%, 50%, 60%, 70% of the training data.

Post-pruning: we perform tree pruning by the reduced error pruning method in the following order: Randomly select 70% of the training data as training data and the remaining 25% as validation data. Then, with the help of the training data, we make the decision tree with ID3. And with validation data, we perform pruning.

C) Finally, this time we select the entire training data as the training data and 25% of the test data randomly as the validation data. Then, with the help of ID3 training data, we build a decision tree and perform pruning with the validation data.
