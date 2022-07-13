import json
import os
import matplotlib.pyplot as plt

class JsonFile:
    def __init__(self, filepath, filename):
        self.filename = filename
        self.filepath = filepath
        self.data = None
        if not os.path.exists(os.path.join(self.filepath, self.filename + '.json')):
            self.data = {"data":dict(), "metadata":dict()}
        else:
            with open(os.path.join(self.filepath, self.filename + '.json'), 'r') as f:
                self.data = json.load(f)
    
    def print(self):
        print(self.data)
    
    def add_entry(self, names, metadata=False):
        if not metadata:
            entry = self.data["data"]
        else:
            entry = self.data["metadata"]
        for name in names:
            if name in entry.keys():
                entry = entry[name]
            else:
                entry[name] = dict()
                entry = entry[name]
                
    def set_value(self, names, value, metadata=False):
        self.add_entry(names[:-1], metadata)
        if not metadata:
            entry = self.data["data"]
        else:
            entry = self.data["metadata"]
        for i in range(len(names)-1):
            entry = entry[names[i]]
        entry[names[-1]] = value
        
    def get_value(self, names, metadata=False):
        if not metadata:
            entry = self.data["data"]
        else:
            entry = self.data["metadata"]
        for i in range(len(names)-1):
            entry = entry[names[i]]
        return entry[names[-1]]
    
    def exists_entry(self, names, metadata=False):
        if not metadata:
            entry = self.data["data"]
        else:
            entry = self.data["metadata"]
        for name in names:
            try:
                entry = entry[name]
            except:
                return False
        if entry:
            return True
        else:
            return False
    
    def save(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath, exist_ok=True)
        with open(os.path.join(self.filepath, self.filename + ".json"), 'w') as f:
            f.write(json.dumps(self.data, indent=4))
    
def save_as_plot(filepath, filename, figure):
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    figure.savefig(os.path.join(filepath, filename + '.png'), transparent=True,
                   bbox_inches='tight')