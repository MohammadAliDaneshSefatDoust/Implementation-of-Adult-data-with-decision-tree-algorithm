import numpy as np
from sklearn.metrics import accuracy_score
    #class Id3class
class Id3class:
  def train(self, input, output):
    self.size_of_tree = 0
    data = input.copy()
    data[output.name] = output
    self.tree = self.build_tree(data, data, input.columns, output.name)
    
  def build_tree(self, data, orginal_data, feature_attribute_names, target, parent_node_class=None):
    self.size_of_tree = self.size_of_tree + 1
    #if data is pure
    unique_classes, counts = np.unique(data[target], return_counts=True)
    if len(unique_classes) <= 1:
      return [int(unique_classes[0]), counts[0]]
    #if subset is empty
    elif len(data) == 0:
      majority_class_index = np.argmax(np.unique(original_data[target], return_counts=True)[1])
      unique_class, counts = np.unique(original_data[target], return_counts=True)[majority_class_index]
      return [int(unique_class), counts]
    #if data set contains no features to train with
    elif len(feature_attribute_names) == 0:
      return [int(parent_node_class), 0]
    #construct a branch:
    else:
      #determine parent node class of current branch
      majority_class_index = np.argmax(np.unique(data[target], return_counts=True)[1])
      parent_node_class = unique_classes[majority_class_index]

      ig_values = [self.information_gain(data, feature, target) for feature in feature_attribute_names]
      best_feature_index = np.argmax(ig_values)
      best_feature = feature_attribute_names[best_feature_index]

      #create tree structure
      tree = {best_feature: {}}

      feature_attribute_names = [i for i in feature_attribute_names if i != best_feature]

      parent_attribute_values = np.unique(data[best_feature])
      for value in parent_attribute_values:
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = self.build_tree(sub_data, orginal_data, feature_attribute_names, target, parent_node_class)
        tree[best_feature][value] = subtree

      return tree

  def entropy(self, attribute_column):
    #find unique values and their frequency counts
    values, counts = np.unique(attribute_column, return_counts=True)
    entropy_list = []

    for i in range(len(values)):
      probability = counts[i]/np.sum(counts)
      entropy_list.append(-probability*np.log2(probability))

    total_entropy = np.sum(entropy_list)

    return total_entropy

  def information_gain(self, data, feature_attribute_name, target):
    total_entropy = self.entropy(data[target])

    values, counts = np.unique(data[feature_attribute_name], return_counts=True)

    weighted_entropy_list = []

    for i in range(len(values)):
      subset_probability = counts[i]/np.sum(counts)
      subset_entropy = self.entropy(data.where(data[feature_attribute_name]==values[i]).dropna()[target])
      weighted_entropy_list.append(subset_probability*subset_entropy)

    total_weighted_entropy = np.sum(weighted_entropy_list)

    #calculate information gain
    information_gain = total_entropy - total_weighted_entropy

    return information_gain


  def predict(self, input, limit_bool, limit):
    samples = input.to_dict(orient='records')
    predictions = []
    for sample in samples:
      predictions.append(self.recursive_prediction(sample, self.tree, 1, limit_bool, limit, 0))
      
    return predictions

  def recursive_prediction(self, sample, tree, default, limit_bool, limit, size):
    counter_node = size
    for attribute in list(tree.keys()):
      for value in list(tree[attribute].keys()):
        counter_node += 1
    
    if limit_bool == True and limit < counter_node:
      return int(default)
    #map sample data to tree
    for attribute in list(sample.keys()):
      if attribute in list(tree.keys()):
        try:
          result = tree[attribute][sample[attribute]]
        except:
          return int(default)

        result = tree[attribute][sample[attribute]]

        #if more attributes exist within result, recursively find best result
        if isinstance(result, dict):
          return self.recursive_prediction(sample, result, default, limit_bool, limit, counter_node)
        else:
          return int(result[0])
  
  def get_size_of_tree(self):
      return self.size_of_tree
  
  def prune(self, input, output):
      self.recursive_prune(self.tree, input, output, None, None)

  def recursive_prune(self, tree, input, output, parent = None, branch = None):
      pruned = True
      
      for attribute in list(tree.keys()):
        for value in list(tree[attribute].keys()):
          sub_tree = tree[attribute][value]
          if isinstance(sub_tree, dict):
            pruned = self.recursive_prune(sub_tree, input, output, tree[attribute], value)
                  
      if parent is None:
        return False
    
      #if all subtree has been pruned
      if pruned == True:
        count_class0, count_class1, count, class_ = 0, 0, 0, 0
        count_nodes = 0
        for atrribute in list(tree.keys()):
          for value in list(tree[attribute].keys()):
            if isinstance(tree[attribute][value], dict):
              return False
            count_nodes += 1
            if tree[attribute][value][0] == 0:
              count_class0 += tree[attribute][value][1]
            elif tree[attribute][value][0] == 1:
              count_class1 += tree[attribute][value][1]
          if count_class0 >= count_class1:
            count = count_class0
            class_ = 0
          else:
            count = count_class1
            class_ = 1
        
        predict_simple = self.predict(input, False, 0)
        accuracy_simple = accuracy_score(output, predict_simple)

        parent_branch_copy = parent[branch].copy()
        
        parent[branch] = [class_, count]
        predict_pruned = self.predict(input, False, 0)
        accuracy_pruned = accuracy_score(output, predict_pruned)
        
        if accuracy_pruned > accuracy_simple + 0.0001:
          self.size_of_tree = self.size_of_tree - count_nodes
          return True
        else:
          parent[branch] = parent_branch_copy
          return False
