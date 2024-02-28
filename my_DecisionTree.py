from abc import ABC, abstractmethod

class DecisionTree(ABC):
    @abstractmethod
    def classify(self, instance):
        """
        Evaluates the learned decision tree on a single instance.

        :return: the classification of the instance
        """
        pass

    @abstractmethod
    def print(self):
        """
        Prints the tree in specified format.
        """
        pass

    @abstractmethod
    def rootInfoGain(self, train):
        """
        Print the information gain of each attribute as computed from creating the root node for the
        given DataSet.

        Print each line with one attribute
        the Attr_name then a space then the info gain use precision of 5
        decimal places in output.

        Example:
        A1 0.12345
        A2 0.45678
        A3 0.24890
        ....
        """
        pass

    @abstractmethod
    def printAccuracy(self, test):
        """
        Print the accuracy of the classification for test set with 5 decimal places
        Example:
        0.12345
        """
        pass