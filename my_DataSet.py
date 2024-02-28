from my_Instance import Instance


class DataSet:
    def __init__(self):
        self.labels = None  # ordered list of class labels
        self.attributes = None  # ordered list of attributes
        self.attribute_values = (
            None  # map to ordered discrete values taken by attributes
        )
        self.instances = None  # ordered list of instances
        self.DELIMITER = ","  # Used to split input strings

    def addLabels(self, line):
        self.labels = []

        splitline = line.split(self.DELIMITER)
        if len(splitline) < 2:
            print("Line doesn't contain enough labels")
            return

        # each element is a label, skip the "%%" string
        for i in range(1, len(splitline)):
            self.labels.append(splitline[i])

    def addAttribute(self, line):
        line = line.strip()  # remove trailing characters

        if self.attributes is None:
            self.attributes = []
            self.attribute_values = {}

        splitline = line.split(self.DELIMITER)
        if len(splitline) < 3:
            print("Line doesn't contain enough attributes")
            return

        list = []

        # grab the attribute name
        self.attributes.append(splitline[1])
        self.attribute_values[splitline[1]] = list

        # ordered list of values for specific attribute
        for i in range(2, len(splitline)):
            list.append(splitline[i])

    def addInstance(self, line):
        line = line.strip()  # remove trailing characters
        if self.instances is None:
            self.instances = []

        splitline = line.split(self.DELIMITER)
        if len(splitline) < 1 + len(self.attributes):
            print("Instance doesn't contain enough attributes")
            return

        instance = Instance()
        instance.label = splitline[len(self.attributes)]

        # add the values, will be input in same order as attributes
        for i in range(len(splitline) - 1):
            values = self.attribute_values[self.attributes[i]]
            # find the index of the value
            value_found = False
            for j in range(len(values)):
                if values[j] == splitline[i]:
                    instance.attributes.append(j)
                    value_found = True
                    break

            if not value_found:
                print(
                    f"Value {splitline[i]} not found in attribute {self.attributes[i]}"
                )
                return

        self.instances.append(instance)

    def sameMetaValues(self, other):
        # compare labels
        if other.labels is None or self.labels is None:
            if not (self.labels is None and other.labels is None):
                return False
        elif len(other.labels) != len(self.labels):
            return False
        else:
            for i in range(len(other.labels)):
                if other.labels[i] != self.labels[i]:
                    return False

        # compare attributes (and values)
        if other.attributes is None or self.attributes is None:
            if not (self.attributes is None and self.attributes is None):
                return False
        elif (
            len(other.attributes) != len(self.attributes)
            or len(other.attributes) != len(other.attribute_values)
            or len(other.attribute_values) != len(self.attribute_values)
        ):
            return False
        else:
            for i in range(len(other.attributes)):
                if other.attributes[i] != self.attributes[i]:
                    return False
                other_values = other.attribute_values[other.attributes[i]]
                this_values = self.attribute_values[other.attributes[i]]
                for j in range(len(other_values)):
                    if other_values[j] != this_values[j]:
                        return False

        return True
