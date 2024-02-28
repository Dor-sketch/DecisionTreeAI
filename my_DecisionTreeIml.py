from my_DecisionTree import DecisionTree
from math import log2
# if you want to print the p in a a/b format, use the following import
# - helped me to understand the fractions and debug the code
from fractions import Fraction
# print(Fraction.from_float(p).limit_denominator())


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


class DecTreeNode:
    def __init__(self, label, attribute, parent_attribute_value, terminal):
        self.label = label
        self.attribute = attribute
        self.parent_attribute_value = parent_attribute_value
        self.terminal = terminal
        self.children = []

    def add_child(self, node):
        self.children.append(node)

    def __str__(self):
        return f"DecTreeNode: label={self.label}, attribute={self.attribute}, parent_attribute_value={self.parent_attribute_value}, terminal={self.terminal}"


class DecisionTreeImpl(DecisionTree):
    def __init__(self, train=None):
        if train is not None:
            self.labels = train.labels
            self.attributes = train.attributes
            self.attribute_values = train.attribute_values
            self.root = self.learnDecisionTree(
                train.instances, train.attributes, [])
        else:
            print("train is None")

    def learnDecisionTree(self, examples, attributes, parent_examples=()):
        # If examples is empty then return plurality value of parent_examples
        if not examples:
            parent_labels = [ex.label for ex in parent_examples]
            pluralityValue = max(set(parent_labels), key=parent_labels.count)
            return DecTreeNode(pluralityValue, None, None, True)

        # If all examples have the same classification, return the classification
        elif all(ex.label == examples[0].label for ex in examples):
            return DecTreeNode(examples[0].label, None, None, True)

        # If attributes is empty, return plurality value of examples
        elif not attributes:
            example_labels = [ex.label for ex in examples]
            return DecTreeNode(
                max(set(example_labels), key=example_labels.count), None, None, True
            )

        else:
            # Choose the best attribute to split on
            best_attribute = self.choose_best_attribute(attributes, examples)
            tree = DecTreeNode(
                None, best_attribute, None, False
            )  # No label for internal nodes

            # For each value of best_attribute, add a new branch to the tree
            for value in set(ex[best_attribute] for ex in examples):
                new_attributes = [a for a in attributes if a != best_attribute]
                new_examples = [
                    ex for ex in examples if ex[best_attribute] == value]
                subtree = self.learnDecisionTree(
                    new_examples, new_attributes, examples)
                subtree.parent_attribute_value = value
                tree.add_child(subtree)

            return tree

    def choose_best_attribute(self, attributes, examples):
        options = {}
        for attribute in attributes:
            options[attribute] = Formulas(self, examples).Gain(
                attribute
            )
            print(f"{attribute} {options[attribute]:.5f}")
        print()
        return max(options, key=options.get)

    def classify(self, instance):
        if self.root is None:
            return "Root is empty"
        return self.classify_node(instance, self.root)

    def rootInfoGain(self, train=None):
        info_gain = {}
        for attribute in train.attributes:
            info_gain[attribute] = Formulas(
                self, train.instances).Gain(attribute)
        for key, value in info_gain.items():
            print(f"{key} {value:.5f}")

    def classify_node(self, instance, node):
        if node.terminal:
            return node.label
        else:
            value = instance[node.attribute]
            for child in node.children:
                if child.parent_attribute_value == value:
                    return self.classify_node(instance, child)
            # no child found with the value of the instance - returning first label as default
            return self.labels[0]

    def printAccuracy(self, test):
        # print only the accuracy of the classification for test set with 5 decimal places
        correct = 0
        for instance in test.instances:
            classification = self.classify(instance)
            if classification == instance.label:
                correct += 1
        accuracy = correct / len(test.instances)
        print(f"{accuracy:.5f}")

    def print_tree(self, node, depth=0, prefix="", is_last=False):
        branch = "└── " if is_last else "├── "
        if node.terminal:
            print(f"{prefix}{branch}Label: {node.label}")
        else:
            print(f"{prefix}{branch}[{node.attribute}=?]")
            prefix += "    " if is_last else "│   "
            for i, child in enumerate(node.children):
                child_value = self.attribute_values[node.attribute][
                    child.parent_attribute_value
                ]
                print(f"{prefix}│ {child_value}")
                self.print_tree(child, depth + 1, prefix,
                                i == len(node.children) - 1)

    def print(self):
        self.print_tree(self.root, 0, is_last=True)

    def __str__(self):
        return f"DecisionTreeImpl: labels={self.labels}, attributes={self.attributes}, attribute_values={self.attribute_values}, root={self.root}"
