import math
import os
import pickle
import time
import uuid
from typing import Any, Annotated, Callable, Dict, List, Literal, Optional, TypedDict, Union
import string
import random
import copy
from src.dag_visualizer import build_dag, generate_html, save_html

from pydantic import BaseModel, Field
from langchain_core.tools import tool

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

def _new_id(prefix: str) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=8))


class NumericArtifact(BaseModel):
    result_id: str
    tool_name: str
    value: float
    args: Dict[str, Any]
    description: str
    reasons: Dict[str, str]
    parent_result_ids_w_args: Dict[str, str | List[str]] = Field(default_factory=dict)
    parent_result_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timeStamp: float = Field(default_factory=lambda: time.time())
    
class OtherArtifact(BaseModel):
    result_id: str
    tool_name: str
    value: Any
    args: Dict[str, Any]
    description: str
    reasons: Dict[str, str]
    parent_result_ids_w_args: Dict[str, str | List[str]] = Field(default_factory=dict)
    parent_result_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timeStamp: float = Field(default_factory=lambda: time.time())
    
class ListedArtifact(BaseModel):
    result_id: str
    tool_name: str
    value: List[Union[NumericArtifact, OtherArtifact]]
    args: Dict[str, Any]
    description: str
    reasons: Dict[str, str]
    parent_result_ids_w_args: Dict[str, str | List[str]] = Field(default_factory=dict)
    parent_result_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timeStamp: float = Field(default_factory=lambda: time.time())


# class CanvasEntry(BaseModel):
#     entry_type: Literal["note", "numerical_result", "special"]
#     value: Any
#     trusted: bool = False
#     source_result_id: Optional[str] = None
#     metadata: Dict[str, Any] = Field(default_factory=dict)

class myCANVAS():
        
    def __init__(self, working_directory = os.getcwd()):
        self.SpecialKeys = ["ready_to_run_job_list", "finished_job_list"]
        self.canvas = {}
        self.canvas_checkpoints = []
        self.working_directory = working_directory
        self.tmpCkpIdx = 0
        # authoritative provenance registry for numeric tool outputs
        self.result_registry = {}
        self.curr_round_result_ids: List[str] = []

        # optional tool-specific validators
        self.tool_validators: Dict[str, Callable[[NumericArtifact, "myCANVAS"], tuple[bool, str]]] = {}
    
    def set_working_directory(self, working_directory, ckp=0):
        self.working_directory = working_directory
        # writeDir = os.path.join(self.working_directory, 'canvas.pickle')
        # ckpDir = os.path.join(self.working_directory, 'canvas_checkpoints.pickle')
        # if not os.path.exists(writeDir):
        #     with open(writeDir, 'wb') as f:
        #         pickle.dump(self.canvas, f)
        #     print("##################### CANVAS #######################")
        #     myDictPP(self.canvas, toDisk=True, filename=writeDir+'.txt')
        #     print("################### CANVAS END #####################")    
        # else:
        #     with open(ckpDir, 'rb') as f:
        #         self.canvas_checkpoints = pickle.load(f)
        #     print(f"loaded {len(self.canvas_checkpoints)} checkpoints from {ckpDir}")
        #     self.canvas = self.canvas_checkpoints[-1-ckp] 
        #     self.tmpCkpIdx = ckp
        #     print("##################### CANVAS #######################")
        #     myDictPP(self.canvas, toDisk=True, filename=writeDir+'.txt')
        #     print("################### CANVAS END #####################")
    
    def print(self):
        print("##################### CANVAS #######################")
        myDictPP(self.canvas)
        print("################### CANVAS END #####################")
    
    def inspect(self):
        return list(self.canvas.keys())
    
    def read(self, key):
        if len(self.canvas) == 0:
            return "Canvas is empty"
        notFoundMsg = f"Key '{key}' not found. Please choose from {list(self.canvas.keys())}"
        
        if key == "finished_job_list":
            # turn list into a dict where the key is the index and the value is the job name
            returning_finished_job_list = {i: job for i, job in enumerate(self.canvas[key].items())}
            return repr(returning_finished_job_list)
        
        return f"{self.canvas.get(key, notFoundMsg)}"
        
    def write(self, key, value, overwrite=False):
        writeDir = os.path.join(self.working_directory, 'canvas.pickle')
        # if key not in self.canvas:
        
        if key in self.SpecialKeys:
            if key == "finished_job_list":
                return f"Key '{key}' is read-only and cannot be overwritten."
            assert isinstance(value, dict), f"Value for key '{key}' must be a dict."
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()), f"All keys and values in the dict for key '{key}' must be strings."
            # assert all(isinstance(t, tuple) and len(t) == 2 and all(isinstance(s, str) for s in t) for t in value), f"All elements in the list for key '{key}' must be tuples of (str(job_name), str(id))."
            
            
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
    
    def snap(self):
        self.canvas_checkpoints.append(copy.deepcopy(self.canvas))

    def snap_save(self):
        if self.tmpCkpIdx > 0:
            self.canvas_checkpoints[-1-self.tmpCkpIdx:-1] = []
            self.tmpCkpIdx = 0
        ckpDir = os.path.join(self.working_directory, 'canvas_checkpoints.pickle')
        with open(ckpDir, 'wb') as f:
                pickle.dump(self.canvas_checkpoints, f)   
                
    def register_tool_validator(
        self,
        tool_name: str,
        validator_fn: Callable[[NumericArtifact, "myCANVAS"], tuple[bool, str]],
    ):
        self.tool_validators[tool_name] = validator_fn

    def register_tool_output(
        self,
        tool_name: str,
        args: Dict[str, Any],
        value: Any,
        description: str,
        listed_value: bool = False,
        reasons: Dict[str, str] = {},
        parent_result_ids: Optional[List[str]] = None,
        parent_result_ids_w_args: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        All tools should regiester their outputs through this method to ensure proper provenance tracking and verification.
        This is the only authoritative source for numeric provenance. if numerical_result == True, type(value) must be int or float
        """
        
        result_id = _new_id("")
        
        try:
            value = float(value)
            artifact = NumericArtifact(
                result_id=result_id,
                tool_name=tool_name,
                args=args,
                value=value,
                description=description,
                reasons=reasons,
                parent_result_ids=parent_result_ids or [],
                parent_result_ids_w_args=parent_result_ids_w_args or {},
                metadata=metadata or {},
            )
            # add duplication check
            self.result_registry[result_id] = artifact
            self.curr_round_result_ids.append(result_id)
            return result_id
        except:
            if listed_value:
                artifactList = []
                for i, v in enumerate(value):
                    try:
                        v = float(v)
                        artifactList.append(NumericArtifact(
                            result_id=result_id,
                            tool_name=tool_name,
                            args=args,
                            value=v,
                            description=description,
                            reasons=reasons,
                            parent_result_ids=parent_result_ids or [],
                            parent_result_ids_w_args=parent_result_ids_w_args or {},
                            metadata=metadata or {},
                        ))
                    except:
                        artifactList.append(OtherArtifact(
                            result_id=result_id,
                            tool_name=tool_name,
                            args=args,
                            value=v,
                            description=description,
                            reasons=reasons,
                            parent_result_ids=parent_result_ids or [],
                            parent_result_ids_w_args=parent_result_ids_w_args or {},
                            metadata=metadata or {},
                        ))
                        
                artifact = ListedArtifact(
                    result_id=result_id,
                    tool_name=tool_name,
                    args=args,
                    value=artifactList,
                    description=description,
                    reasons=reasons,
                    parent_result_ids=parent_result_ids or [],
                    parent_result_ids_w_args=parent_result_ids_w_args or {},
                    metadata=metadata or {},
                )
                self.result_registry[result_id] = artifact
                self.curr_round_result_ids.append(result_id)
                return result_id
            else:
                artifact = OtherArtifact(
                    result_id=result_id,
                    tool_name=tool_name,
                    args=args,
                    value=value,
                    description=description,
                    reasons=reasons,
                    parent_result_ids=parent_result_ids or [],
                    parent_result_ids_w_args=parent_result_ids_w_args or {},
                    metadata=metadata or {},
                )
                self.result_registry[result_id] = artifact
                self.curr_round_result_ids.append(result_id)
                return result_id
        
    def get_artifact(self, result_id: str):
        return self.result_registry.get(result_id, None)
    
    def verify_artifact(
        self,
        expected_value: Any,
        source_result_id,
        tol: float = 1e-10,
    ) -> tuple[bool, str, Optional[NumericArtifact]]:
        
        if source_result_id == "PLACEHOLDER":
            return True, "Verification success."
        if len(source_result_id) != 8:
            return False, f"Invalid result ID format: Tool generated ID would be an 8-character string. Did you mean 'PLACEHOLDER'?"

        artifact = self.result_registry.get(source_result_id)
        if artifact is None:
            return False, f"ID {source_result_id} does not exist."

        if isinstance(artifact, ListedArtifact):
            # as long as one of the artifacts in the list matches the expected value, we consider it a success
            for sub_artifact in artifact.value:
                if isinstance(sub_artifact.value, (int, float)):
                    try:
                        expected_value = float(expected_value)
                        if math.isclose(float(expected_value), sub_artifact.value, rel_tol=0.0, abs_tol=tol):
                            return True, f"{source_result_id} Verification success."
                    except ValueError:
                        continue
                else:
                    if expected_value == sub_artifact.value:
                        return True, f"{source_result_id} Verification success."
            return False, f"ID {source_result_id} listed verification failed. \nExpected value: {repr(expected_value)} does not match any of the registered tool outputs {[repr(sub_artifact.value) for sub_artifact in artifact.value]}."
        else:
            if isinstance(artifact.value, (int, float)):
                try:
                    expected_value = float(expected_value)
                    if not math.isclose(float(expected_value), artifact.value, rel_tol=0.0, abs_tol=tol):
                        return False, f"ID {source_result_id} verification failed. \nExpected value: {repr(expected_value)} does not match registered tool output {repr(artifact.value)}."
                except ValueError:
                    return False, f"ID {source_result_id} verification failed. \nExpected value: {repr(expected_value)} does not match registered tool output {repr(artifact.value)}."
            else:
                if expected_value != artifact.value:
                    return False, f"ID {source_result_id} verification failed. \nExpected value: {repr(expected_value)} does not match registered tool output {repr(artifact.value)}."
                
            return True, f"{source_result_id} Verification success."

        
    def rest_curr_round_result_ids(self):
        self.curr_round_result_ids = []
        
    def check_required_tool_use(self, required_tools: str):
        # missing_tools = set(required_tools) - set([self.get_artifact(result_id).tool_name for result_id in self.curr_round_result_ids])
        # # remove "" from missing tools if it exists
        # missing_tools = [tool for tool in missing_tools if tool != ""]
        # if missing_tools:
        if required_tools not in [self.get_artifact(result_id).tool_name for result_id in self.curr_round_result_ids]:
            return False, f"{required_tools}"
        else:
            return True, "All required tools have been used in this round."
        
    def gen_DAG(self, filename, title):
        my_artifact_nodes = [self.get_artifact(result_id) for result_id in self.result_registry]
        dag  = build_dag(my_artifact_nodes)
        html = generate_html(dag, title=title)
        save_html(html, filename)
        
    
CANVAS = myCANVAS()

# if __name__ == "__main__":
#     print(CANVAS.inspect())
#     print(CANVAS.write('test', 'test value3'))
#     print(CANVAS.write('test2', 'test2 value3', overwrite=True))
#     tmp = read("/nfs/turbo/coe-venkvis/ziqiw-turbo/material_agent/Rb-BCC-plan/Rb_bcc_k_0.5_ecutwfc_70.in.pwo")
#     print(CANVAS.write('Rb_bcc_k_0.5_ecutwfc_70', tmp))
#     dd = {'a': 1, 'b': 2}
#     ddd = {'c': 3, 'd': 4, 'dd': dd}
#     ll = [1, 2, 3, 4, dd, ddd]
#     print(CANVAS.write('dict', dd))
#     print(CANVAS.write('dict2', ddd))
#     print(CANVAS.write('list', ll))
#     print(CANVAS.inspect())

#     loaded = pickle.load(open('canvas.pickle', 'rb'))
#     myDictPP(pickle.load(open('canvas.pickle', 'rb')))
#     print(loaded == CANVAS.canvas)
    
    

    
