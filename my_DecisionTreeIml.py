from my_DecisionTree import DecisionTree
# if you want to print the p in a a/b format, use the following import
# - helped me to understand the fractions and debug the code
from fractions import Fraction
# print(Fraction.from_float(p).limit_denominator())

# if you want to print the tree building process, use the following imports
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Formulas import Formulas


node_count = 0



class DecTreeNode:
    def __init__(self, label, attribute, parent_attribute_value, terminal):
        global node_count # used to give each node a unique id, for animation purposes
        self.label = label
        self.attribute = attribute
        self.parent_attribute_value = parent_attribute_value
        self.terminal = terminal
        self.children = []
        self.id = node_count
        node_count += 1

    def add_child(self, node):
        self.children.append(node)

    def __str__(self):
        ret = f"{self.id} "
        if self.terminal:
            ret += f"Leaf({self.label})"
        else:
            ret += f"Node({self.attribute}=?)"
        return ret


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

    def parse_tree(self, node, depth=0, prefix="", is_last=False):
        """
        return node list and edge list for the networkx graph
        """

        if depth == 0:
            self.node_list = []
            self.edge_list = []

        self.node_list.append(str(node))
        if node.terminal:
            return
        else:
            for i, child in enumerate(node.children):
                self.edge_list.append((str(node), str(child)))
                self.parse_tree(child, depth + 1, prefix, i == len(node.children) - 1)

        return self.node_list, self.edge_list


    def animate_building_tree(self):
        """
        Generate a gif of the tree building process
        """

        node_list, edge_list = self.parse_tree(self.root)
        tree = nx.DiGraph()

        for node in node_list:
            tree.add_node(node)
        for edge in edge_list:
            tree.add_edge(edge[0], edge[1])

        # Create a layout for nodes
        pos = nx.spring_layout(tree)
        fig, ax = plt.subplots()

        def update(num):
            ax.clear()
            # Create a new graph for each frame
            G = nx.DiGraph()
            # Add nodes and edges up to the current frame number
            for node in node_list[:num+1]:
                G.add_node(node)
            for edge in edge_list[:num]:
                G.add_edge(edge[0], edge[1])
            # Draw the graph
            nx.draw(G, pos, with_labels=True, ax=ax, node_color='lightblue',
                    edgelist=list(G.edges()), font_size=6)
            ax.set_title(f"Building Tree: {num}/{len(edge_list)} edges")


        ani = animation.FuncAnimation(fig, update, frames=range(
            len(edge_list)+1), repeat=True, interval=10)

        plt.show()

        ani.save('building_tree.gif', writer='imagemagick', fps=6)