import pickle
from ase.io import read
import os


def myDictPP(myDict, indent=4, nindent=0, toDisk=False, filename=None):
    # remove canvas.txt if it exists
    if toDisk:
        with open(filename, 'w') as f:
            f.write("")
    _myDictPP(myDict, indent, nindent, toDisk, filename)

def _myDictPP(myDict, indent=4, nindent=0, toDisk=False, filename=None):
    print('{')
    if toDisk:
        with open(filename, 'a') as f:
            f.write("{\n")
    for k, v in myDict.items():
        print(" " * indent * nindent, end="")
        if toDisk:
            with open(filename, 'a') as f:
                f.write(" " * indent * nindent)
        if isinstance(v, dict):
            print(repr(k) + ": ", end="")
            if toDisk:
                with open(filename, 'a') as f:
                    f.write(repr(k) + ": ")
            _myDictPP(v, indent, nindent+1, toDisk=toDisk, filename=filename)
        else:
            print(repr(k) + ": " + repr(v) + ",")
            if toDisk:
                with open(filename, 'a') as f:
                    f.write(repr(k) + ": " + repr(v) + ",\n")
    print(" " * indent * nindent + "}")
    if toDisk:
        with open(filename, 'a') as f:
            f.write(" " * indent * nindent + "}\n")

class myCANVAS():
        
    def __init__(self, working_directory = os.getcwd()):
        self.SpecialKeys = ["ready_to_run_job_list", "finished_job_list"]
        self.canvas = {}
        self.working_directory = working_directory
    
    def set_working_directory(self, working_directory):
        self.working_directory = working_directory
    
    def inspect(self):
        return list(self.canvas.keys())
    
    def read(self, key):
        if len(self.canvas) == 0:
            return "Canvas is empty"
        notFoundMsg = f"Key '{key}' not found. Please choose from {list(self.canvas.keys())}"
        return f"{self.canvas.get(key, notFoundMsg)}"
        
    def write(self, key, value, overwrite=False):
        writeDir = os.path.join(self.working_directory, 'canvas.pickle')
        # if key not in self.canvas:
        
        if key in self.SpecialKeys:
            if key == "finished_job_list":
                return f"Key '{key}' is read-only and cannot be overwritten."
            assert isinstance(value, list), f"Value for key '{key}' must be a list."
            assert all(isinstance(i, str) for i in value), f"All elements in the list for key '{key}' must be strings of job names."
            
        if key not in self.canvas.keys():
            self.canvas[key] = value
            with open(writeDir, 'wb') as f:
                pickle.dump(self.canvas, f)
            print("##################### CANVAS #######################")
            myDictPP(self.canvas, toDisk=True, filename=writeDir+'.txt')
            print("################### CANVAS END #####################")    
            return f"Key '{key}' successfully added."
        elif overwrite:
            self.canvas[key] = value
            with open(writeDir, 'wb') as f:
                pickle.dump(self.canvas, f)    
            print("##################### CANVAS #######################")
            myDictPP(self.canvas, toDisk=True, filename=writeDir+'.txt')
            print("################### CANVAS END #####################")   
            return f"Key '{key}' successfully overwritten."
        else:
            return f"Key '{key}' already exists. Please choose a different key. If you want to overwrite the value, set the 'overwrite' flag to True."
    
CANVAS = myCANVAS()

if __name__ == "__main__":
    print(CANVAS.inspect())
    print(CANVAS.write('test', 'test value3'))
    print(CANVAS.write('test2', 'test2 value3', overwrite=True))
    tmp = read("/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/Rb-BCC-plan/Rb_bcc_k_0.5_ecutwfc_70.in.pwo")
    print(CANVAS.write('Rb_bcc_k_0.5_ecutwfc_70', tmp))
    dd = {'a': 1, 'b': 2}
    ddd = {'c': 3, 'd': 4, 'dd': dd}
    ll = [1, 2, 3, 4, dd, ddd]
    print(CANVAS.write('dict', dd))
    print(CANVAS.write('dict2', ddd))
    print(CANVAS.write('list', ll))
    print(CANVAS.inspect())

    loaded = pickle.load(open('canvas.pickle', 'rb'))
    myDictPP(pickle.load(open('canvas.pickle', 'rb')))
    print(loaded == CANVAS.canvas)
    
    

    
