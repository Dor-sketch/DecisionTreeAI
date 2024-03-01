from math import log2
from fractions import Fraction


class Formulas:
    def __init__(self, decision_tree, training_set):
        if decision_tree is None:
            decision_tree = DecisionTreeImpl(training_set)
        self.attributes = decision_tree.attributes
        self.training_set = training_set
        self.attribute_values = decision_tree.attribute_values
        self.labels = decision_tree.labels

    def B(self, q):
        """Return the entropy of a Boolean random variable
        that is true with probability q and false
        with probability 1-q."""
        if q == 0 or q == 1:
            return 0
        return -q * log2(q) - (1 - q) * log2(1 - q)

    def P(self, attribute, value):
        """Return the probability of the attribute equaling a specific value."""
        att_count = 0
        # important: use the index of the value in the attribute_values list
        value_index = self.attribute_values[attribute].index(value)
        for instance in self.training_set:
            if instance[attribute] == value_index:
                att_count += 1
        total_count = len(self.training_set)
        return att_count / total_count if total_count > 0 else 0

    def H(self, attribute):
        """Return the entropy of the attribute."""
        if attribute not in self.attributes:
            return 0

        entropy = 0
        for value in self.attribute_values[attribute]:
            # Probability of this attribute value in the dataset
            p = self.P(attribute, value)
            if p > 0:  # To avoid log2(0) which is undefined
                entropy += -p * log2(p)  # Apply the entropy formula

        return entropy

    def Remainder(self, attribute):
        """Return the remainder of the attribute."""
        reminder = 0
        subSets = {}
        for instance in self.training_set:
            key = instance[attribute]
            if key not in subSets:
                subSets[key] = []
            subSets[key].append(instance)
        total = len(self.training_set)
        for subSet in subSets.values():
            p = sum(1 for instance in subSet if instance.label ==
                    self.labels[0])
            n = len(subSet) - p
            if (p + n) > 0:
                reminder += (p + n) / total * self.B(p / (p + n))
        return reminder

    def classificationH(self):
        """Return the entropy of the label"""
        entropy = 0
        for label in self.labels:
            p = sum(1 for instance in self.training_set if instance.label == label)
            p /= len(self.training_set)
            print(f"{sum(1 for instance in self.training_set if instance.label == label)} / {len(self.training_set)}")
            print(Fraction(1-p).limit_denominator())
            if p > 0:
                entropy += -p * log2(p) - (1 - p) * log2(1 - p)
        return entropy

    def Gain(self, attribute):
        """Return the gain of the attribute."""
        print(f"classificationH: {self.classificationH()}")
        print(f"Remainder({attribute}): {self.Remainder(attribute)}")
        print(f"Gain({attribute}): {self.classificationH() - self.Remainder(attribute)}")
        return self.classificationH() - self.Remainder(attribute)
