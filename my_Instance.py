class Instance:
    def __init__(self):
        self.label = None
        self.attributes = []

    def addAttribute(self, i):
        self.attributes.append(i)

    def __str__(self):
        attributes = ", ".join(str(i) for i in self.attributes)
        return f"Instance: label={self.label}, attributes={attributes}"


    def __getitem__(self, key):
        return self.attributes[int(key[1:])-1]

    def __setitem__(self, key, value):
        self.attributes[key] = value