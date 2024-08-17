import numpy as np
from numpy import ndarray

from .data_types import BestSplit
from .node import Node

class DecisionTree:
    """
    A decision tree classifier for binary classification problems.
    """

    def __init__(self, min_samples:int=2, max_depth:int=2):
        """
        A decision tree classifier for binary classification problems.
        """
        self.min_samples = min_samples
        self.max_depth = max_depth

    def split_dataset(self, dataset: ndarray, feature: int, threshold: float) -> tuple[ndarray, ndarray]:
        """
        Splits the given dataset into two datasets based on the given feature and threshold.

        Parameters:
            dataset (ndarray): Input dataset.
            feature (int): Index of the feature to be split on.
            threshold (float): Threshold value to split the feature on.

        Returns:
            left_dataset (ndarray): Subset of the dataset with values less than or equal to the threshold.
            right_dataset (ndarray): Subset of the dataset with values greater than the threshold.
        """

        left_dataset = []
        right_dataset = []

        for row in dataset:
            if row[feature]<=threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)

        return left_dataset, right_dataset
    
    def entropy(self, y: ndarray) -> float:
        """
        Computes the entropy of the given label values.

        Parameters:
            y (ndarray): Input label values.

        Returns:
            entropy (float): Entropy of the given label values.
        """

        entropy: float = 0
        labels: ndarray = np.unique(y)

        for label in labels:
            # get all rows with label
            label_rows: ndarray = y[y==label]

            label_probability: float = len(label_rows)/len(y)

            entropy += -label_probability*np.log2(label_probability)

        return entropy
    
    def information_gain(self, parent: ndarray, left: ndarray, right:ndarray) -> float:
        """
        Computes the information gain from splitting the parent dataset into two datasets.

        Parameters:
            parent (ndarray): Input parent dataset.
            left (ndarray): Subset of the parent dataset after split on a feature.
            right (ndarray): Subset of the parent dataset after split on a feature.

        Returns:
            information_gain (float): Information gain of the split.
        """
        parent_entropy: float = self.entropy(parent)

        left_ratio: float = len(left)/len(parent)
        right_ratio: float = len(right)/len(parent)

        left_entropy: float = self.entropy(left)
        right_entropy: float = self.entropy(right)

        child_weighted_entropy: float = left_ratio*left_entropy + right_ratio*right_entropy

        information_gain: float = parent_entropy - child_weighted_entropy

        return information_gain
    
    def best_split(self, dataset: ndarray, num_samples: int, num_features: int) -> BestSplit:
        """
        Finds the best split for the given dataset.

        Args:
        dataset (ndarray): The dataset to split.
        num_samples (int): The number of samples in the dataset.
        num_features (int): The number of features in the dataset.

        Returns:
        dict: A dictionary with the best split feature index, threshold, gain, 
              left and right datasets.
        """
        best_split: BestSplit = {
            "gain": -1,
            "feature": None,
            "left_dataset": None,
            "right_dataset": None,
            "threshold": None
        }

        for feature_index in range(num_features):
            feature_values: ndarray = dataset[:, feature_index]
            thresholds: ndarray = np.unique(feature_values)

            for threshold in thresholds:
                left_dataset, right_dataset = self.split_dataset(dataset, feature_index, threshold)

                if len(left_dataset)!=0 and len(right_dataset)!=0:    
                    y = dataset[:, -1]
                    left_y = left_dataset[:, -1]
                    right_y = right_dataset[:, -1]

                    information_gain = self.information_gain(y, left_y, right_y)
                    if information_gain>best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain

        return best_split
    
    def calculate_leaf_value(self, y) -> int:
        """
        Calculates the most occurring value in the given list of y values.

        Args:
            y (list): The list of y values.

        Returns:
            The most occurring value in the list.
        """
        y = list(y)
        #get the highest present class in the array
        most_occuring_value: int = max(y, key=y.count)
        return most_occuring_value
    
    def build_tree(self, dataset: ndarray, current_depth:int=0) -> Node:
        """
        Recursively builds a decision tree from the given dataset.

        Args:
        dataset (ndarray): The dataset to build the tree from.
        current_depth (int): The current depth of the tree.

        Returns:
        Node: The root node of the built decision tree.
        """
        x, y = dataset[:, :-1], dataset[:, -1]

        n_samples, n_features = x.shape

        if n_samples>=self.min_samples and current_depth<=self.max_depth:

            best_split = self.best_split(x, n_samples, n_features)

            if best_split["gain"]!=0:
                left_node = self.build_tree(best_split["left_dataset"], current_depth+1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth+1)
                return Node(
                    feature=best_split["feature"], 
                    threshold=best_split["threshold"], 
                    left=best_split["left_dataset"],
                    right=best_split["right_dataset"],
                    gain=best_split["gain"]
                )
            
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def fit(self, x: ndarray, y: ndarray) -> None:
        """
        Builds and fits the decision tree to the given X and y values.

        Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target values.
        """
        dataset = np.concatenate((x, y), axis=1)  
        self.root = self.build_tree(dataset)

    def make_prediction(self, x, node):
        """
        Traverses the decision tree to predict the target value for the given feature vector.

        Args:
        x (ndarray): The feature vector to predict the target value for.
        node (Node): The current node being evaluated.

        Returns:
        The predicted target value for the given feature vector.
        """
        # if the node has value i.e it's a leaf node extract it's value
        if node.value != None: 
            return node.value
        else:
            #if it's node a leaf node we'll get it's feature and traverse through the tree accordingly
            feature = x[node.feature]
            if feature <= node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)

    def predict(self, X: ndarray) -> ndarray:
        """
        Predicts the class labels for each instance in the feature matrix X.

        Args:
        X (ndarray): The feature matrix to make predictions for.

        Returns:
        list: A list of predicted class labels.
        """
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        for x in X:
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions