import math
import os
import pickle
import time
import uuid
from typing import Any, Annotated, Callable, Dict, List, Literal, Optional, TypedDict
import string
import random

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
    parent_result_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class OtherArtifact(BaseModel):
    result_id: str
    tool_name: str
    value: Any
    parent_result_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CanvasEntry(BaseModel):
    entry_type: Literal["note", "numerical_result", "special"]
    value: Any
    trusted: bool = False
    source_result_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class myCANVAS:
    def __init__(self, working_directory=os.getcwd()):
        self.SpecialKeys = ["ready_to_run_job_list", "finished_job_list"]
        self.canvas: Dict[str, CanvasEntry] = {}
        self.canvas_checkpoints = []
        self.working_directory = working_directory
        self.tmpCkpIdx = 0

        # authoritative provenance registry for numeric tool outputs
        self.result_registry: Dict[str, NumericArtifact] = {}

        # optional tool-specific validators
        self.tool_validators: Dict[str, Callable[[NumericArtifact, "myCANVAS"], tuple[bool, str]]] = {}

    def set_working_directory(self, working_directory, ckp=0):
        self.working_directory = working_directory
        
    def get(self, name, notFoundReturns):
        entry = self.canvas.get(name, None)
        if entry is None:
            return notFoundReturns
        return entry.value
    
    def __setattr__(self, name, value):
        entry = self.canvas.get(name, None)
        if entry is None:
            self.canvas[name] = CanvasEntry(
                entry_type="special",
                value=value,
                trusted=True,
                source_result_id=None,
                metadata={},
            )
            self._persist()
            return f"Special key '{name}' successfully updated."
        else:
            entry.value = value
            self._persist()
            return f"Special key '{name}' successfully updated."

    def _persist(self):
        write_dir = os.path.join(self.working_directory, "canvas.pickle")
        payload = {
            "canvas": {k: v.model_dump() for k, v in self.canvas.items()},
            "result_registry": {k: v.model_dump() for k, v in self.result_registry.items()},
        }
        with open(write_dir, "wb") as f:
            pickle.dump(payload, f)

        print("##################### CANVAS #######################")
        myDictPP(payload, toDisk=True, filename=write_dir + ".txt")
        print("################### CANVAS END #####################")

    def print(self):
        print("##################### CANVAS #######################")
        myDictPP(
            {
                "canvas": {k: v.model_dump() for k, v in self.canvas.items()},
                "result_registry": {k: v.model_dump() for k, v in self.result_registry.items()},
            }
        )
        print("################### CANVAS END #####################")

    def _get_key_list(self):
        noteKeys = []
        resultKeys = []
        for key, entry in self.canvas.items():
            if entry.entry_type == "note":
                noteKeys.append(key)
            else:
                resultKeys.append(key)
        
        return f"\nFollowing keys are un-verified notes:\n{noteKeys}\n\nFollowing keys are verified numerical results:\n{resultKeys}"
    
    def inspect(self):
        if len(self.canvas) == 0:
            return "Canvas is empty"
        return self._get_key_list()
        
    def read(self, key):
        if len(self.canvas) == 0:
            return "Canvas is empty"

        if key not in self.canvas:
            # list notes and numerical results separately for easier readability
            return f"Key '{key}' not found. Please choose from: {self._get_key_list()}"

        entry = self.canvas[key]

        if key == "finished_job_list":
            returning_finished_job_list = {i: job for i, job in enumerate(entry.value)}
            return repr(returning_finished_job_list)

        if entry.entry_type == "note":
            # maybe not as dramatic
            return f"below is your note about {key}:\n{entry.value}\n Notes are memory only. Final numerical claims must come from numerical_result entries."

        if entry.entry_type == "numerical_result":
            # just say it's verified and trusted
            return f"below is your verified numerical result for {key}:\n{entry.value}\n Numerical results are trusted. Feel free to use them in further tools calls and final numerical claims"

        return f"{entry.value}"

    # def read_trusted_numerical_result(self, key: str):
    #     if key not in self.canvas:
    #         return f"Key '{key}' not found. Please choose from {list(self.canvas.keys())}"

    #     entry = self.canvas[key]
    #     if entry.entry_type != "numerical_result":
    #         return f"Key '{key}' is not a numerical_result."
    #     if not entry.trusted:
    #         return f"Key '{key}' exists but is not trusted."

    #     return {
    #         "key": key,
    #         "value": float(entry.value),
    #         "source_result_id": entry.source_result_id,
    #         "metadata": entry.metadata,
    #     }

    # def inspect_trusted_numerical_results(self):
    #     out = {}
    #     for key, entry in self.canvas.items():
    #         if entry.entry_type == "numerical_result" and entry.trusted:
    #             out[key] = {
    #                 "value": float(entry.value),
    #                 "source_result_id": entry.source_result_id,
    #                 "metadata": entry.metadata,
    #             }
    #     return out

    def register_tool_validator(
        self,
        tool_name: str,
        validator_fn: Callable[[NumericArtifact, "myCANVAS"], tuple[bool, str]],
    ):
        self.tool_validators[tool_name] = validator_fn

    def register_tool_output(
        self,
        tool_name: str,
        value: Any,
        numerical_result: bool = True,
        parent_result_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        All tools should regiester their outputs through this method to ensure proper provenance tracking and verification.
        This is the only authoritative source for numeric provenance. if numerical_result == True, type(value) must be int or float
        """
        assert not numerical_result or isinstance(value, (int, float)), "For numerical_result=True, value must be int or float."
        
        result_id = _new_id("")
        if numerical_result:
            artifact = NumericArtifact(
                result_id=result_id,
                tool_name=tool_name,
                value=float(value),
                parent_result_ids=parent_result_ids or [],
                metadata=metadata or {},
            )
            # add duplication check
            self.result_registry[result_id] = artifact
            self._persist()
            return result_id
        else:
            artifact = OtherArtifact(
                result_id=result_id,
                tool_name=tool_name,
                value=value,
                parent_result_ids=parent_result_ids or [],
                metadata=metadata or {},
            )
            self.result_registry[result_id] = artifact
            self._persist()
            return result_id
            

    def get_numeric_artifact(self, result_id: str) -> Optional[NumericArtifact]:
        return self.result_registry.get(result_id)
    
    def get_other_artifact(self, result_id: str) -> Optional[OtherArtifact]:
        return self.result_registry.get(result_id)

    def _verify_artifact_recursive(self, result_id: str, visited: Optional[set] = None) -> tuple[bool, str]:
        if visited is None:
            visited = set()

        if result_id in visited:
            return False, f"Cyclic provenance detected at result_id='{result_id}'."
        visited.add(result_id)

        artifact = self.result_registry.get(result_id)
        if artifact is None:
            return False, f"Unknown result_id='{result_id}'."

        for parent_id in artifact.parent_result_ids:
            ok, msg = self._verify_artifact_recursive(parent_id, visited)
            if not ok:
                return False, (
                    f"Result '{result_id}' produced by tool '{artifact.tool_name}' depends on "
                    f"untrusted parent '{parent_id}'. {msg}"
                )

        validator = self.tool_validators.get(artifact.tool_name)
        if validator is not None:
            ok, msg = validator(artifact, self)
            if not ok:
                return False, (
                    f"Tool-specific validation failed for result '{result_id}' "
                    f"(tool='{artifact.tool_name}'). {msg}"
                )

        return True, f"result_id='{result_id}' is recursively trusted."

    def verify_numeric_result_write(
        self,
        expected_value: float,
        source_result_id: Optional[str],
        tol: float = 1e-10,
    ) -> tuple[bool, str, Optional[NumericArtifact]]:
        if source_result_id is None:
            return (
                False,
                "For type='numerical_result', source_result_id is required. "
                "Use the result_id returned by a numerical tool.",
                None,
            )

        artifact = self.result_registry.get(source_result_id)
        if artifact is None:
            return (
                False,
                f"source_result_id='{source_result_id}' was not found in the numeric provenance registry.",
                None,
            )

        if not math.isclose(float(expected_value), artifact.value, rel_tol=0.0, abs_tol=tol):
            return (
                False,
                f"Canvas value {float(expected_value)} does not match registered tool output "
                f"{artifact.value} for source_result_id='{source_result_id}'.",
                None,
            )

        ok, msg = self._verify_artifact_recursive(source_result_id)
        if not ok:
            return False, msg, None

        return True, "Verification passed.", artifact

    def missing_required_numerical_outcomes(self, required_keys: List[str]) -> List[str]:
        missing = []
        for key in required_keys:
            entry = self.canvas.get(key)
            if entry is None:
                missing.append(key)
                continue
            if entry.entry_type != "numerical_result":
                missing.append(key)
                continue
            if not entry.trusted:
                missing.append(key)
        return missing

    def write(
        self,
        key,
        value,
        entry_type: Literal["note", "numerical_result"] = "note",
        overwrite: bool = False,
        source_result_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if key in self.SpecialKeys:
            if key == "finished_job_list":
                return f"Key '{key}' is read-only and cannot be overwritten."

            if not isinstance(value, list):
                return f"Value for key '{key}' must be a list."
            if not all(isinstance(i, str) for i in value):
                return f"All elements in the list for key '{key}' must be strings of job names."

            self.canvas[key] = CanvasEntry(
                entry_type="special",
                value=value,
                trusted=True,
                source_result_id=None,
                metadata={},
            )
            self._persist()
            return f"Special key '{key}' successfully updated."

        if entry_type == "note":
            new_entry = CanvasEntry(
                entry_type="note",
                value=value,
                trusted=False,
                source_result_id=None,
                metadata=metadata or {},
            )

        elif entry_type == "numerical_result":
            if not isinstance(value, (int, float)):
                return (
                    "For entry_type='numerical_result', value must be a float or int. "
                    "Use entry_type='note' for descriptive text."
                )

            ok, msg, artifact = self.verify_numeric_result_write(
                expected_value=float(value),
                source_result_id=source_result_id,
            )
            if not ok:
                # fall back to writing in the note should not be allowed
                return (
                    "NUMERICAL_RESULT_VERIFICATION_FAILED\n"
                    f"{msg}\n"
                    "Please make sure you used correct result from the tools output"
                    "If you didn't use a tool, you must use tool to obtain numerical results! Redo the step with a tool, and then try enter the value again."
                )

            new_entry = CanvasEntry(
                entry_type="numerical_result",
                value=float(value),
                trusted=True,
                source_result_id=source_result_id,
                metadata={
                    "tool_name": artifact.tool_name,
                    **artifact.metadata,
                    **(metadata or {}),
                },
            )

        else:
            return "entry_type must be either 'note' or 'numerical_result'."

        if key not in self.canvas:
            self.canvas[key] = new_entry
            self._persist()
            return f"Key '{key}' successfully added as {entry_type}."

        if overwrite:
            self.canvas[key] = new_entry
            self._persist()
            return f"Key '{key}' successfully overwritten as {entry_type}."

        return (
            f"Key '{key}' already exists. Please choose a different key. "
            "If you want to overwrite the value, set overwrite=True."
        )