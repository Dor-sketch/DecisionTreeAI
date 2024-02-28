import sys
from my_DecisionTreeIml import DecisionTreeImpl
from my_DataSet import DataSet

class HW3:
    @staticmethod
    def parseArgs(args):
        if len(args) < 3:
            print("usage: python3 HW3.py <modeFlag: 0, 1, 2, 3> <trainFilename> "
                  + "<testFilename>")
            # I removed the optional tuneFilename argument since it was not
            # part of the original assignment
            sys.exit(-1)

        mode = int(args[0])
        if 0 > mode or mode > 3:
            print("mode must be between 0 and 3")
            sys.exit(-1)

        return mode, args[1], args[2]

    @staticmethod
    def main(args):
        mode, train_file, test_file = HW3.parseArgs(args)

        trainSet = HW3.createDataSet(train_file)
        tree = DecisionTreeImpl(trainSet)

        if mode == 0:
            # mode 0 : output the mutual information of each attribute at the root node
            tree.rootInfoGain(trainSet)
            return


        if mode == 1:
            # mode 1 : create a decision tree from a training set, output the tree
            tree.print()
            return

        testSet = HW3.createDataSet(test_file)
        if not trainSet.sameMetaValues(testSet):
            print("bad meta-values in test set")
            sys.exit(-1)

        if mode == 2:
            # mode 2 : create a decision tree from a training set, output the classifications of a test set
            for instance in testSet.instances:
                isCorrect = tree.classify(instance) == instance.label
                print(f"Classification of {instance} = {tree.classify(instance)} - Correct? {isCorrect}")
        elif mode == 3:
            # mode 3 : create a decision tree from a training set, output the accuracy of the tree on a test set
            tree.printAccuracy(testSet)

    @staticmethod
    def createDataSet(file):
        my_set = DataSet()
        with open(file, 'r') as f:
            for line in f:
                prefix = line[:2]
                if prefix == "//":
                    continue
                elif prefix == "%%":
                    my_set.addLabels(line)
                elif prefix == "##":
                    my_set.addAttribute(line)
                else:
                    my_set.addInstance(line)
        return my_set

if __name__ == "__main__":
    HW3.main(sys.argv[1:])
