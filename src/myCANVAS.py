import math
import os
import pickle
import re
import time
import uuid
from typing import Any, Annotated, Callable, Dict, List, Literal, Optional, TypedDict, Union
import string
import random
import copy
from src.dag_visualizer import build_dag, generate_html, save_html
from src import var
import numpy as np

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


# Trailing version suffix recognized/managed by the version-controlled
# canvas-doc machinery: '<base>_v<N>' with N >= 1.
_VERSION_SUFFIX_RE = re.compile(r"^(?P<base>.+)_v(?P<num>\d+)$")

# Key prefixes reserved for framework-written content. Agents may read
# these keys but may never create or overwrite them via write().
RESERVED_KEY_PREFIXES = ("JUDGE::", "ARCHIVE::")


# =========================================================
# Debug helper (local copy; myCANVAS cannot import from safety_guard /
# planNexe2 without a circular import). Mirrors their _dbg: prints
# "[DBG][Ns] <msg>", appends to his.txt when var.my_SAVE_DIALOGUE is on,
# and sleeps (length-scaled, capped) so the live log is readable.
# =========================================================

_DBG_ENABLED = True       # master switch for all _dbg output in this module
_DBG_SLEEP = 2            # base factor for the read-pacing sleep
_DBG_SLEEP_MAX = 6        # hard cap (seconds)


def _dbg(msg: str) -> None:
    if not _DBG_ENABLED:
        return
    body = f"[DBG] {msg}"
    sleep_time = min(_DBG_SLEEP_MAX, max(1, int(len(body) * 0.01 * _DBG_SLEEP)))
    outStr = f"[DBG][{sleep_time}s] {msg}"
    print(outStr, flush=True)
    if getattr(var, "my_SAVE_DIALOGUE", False):
        try:
            with open(f"{var.my_WORKING_DIRECTORY}/his.txt", "a") as f:
                f.write(outStr + "\n")
        except Exception:
            pass
    time.sleep(sleep_time)  # ACTIVE sleep so messages can be read live


class NumericArtifact(BaseModel):
    result_id: str
    tool_name: str
    value: float
    args: Dict[str, Any]
    description: str
    context: str = Field(default_factory=str)
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
    context: str = Field(default_factory=str)
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
    context: str = Field(default_factory=str)
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
    
    # ------------------------------------------------------------------
    # Version-controlled doc helpers
    #
    # Convention: a "doc family" is a base key plus zero or more
    # version-suffixed keys. The PLAIN base key counts as version 1.
    # e.g. family 'notes' = {'notes' (v1), 'notes_v2', 'notes_v3'}.
    # Version-suffixed keys can ONLY be created by edit_doc() — write()
    # rejects them — so the suffix space is fully owned by the framework
    # and existing content is never mutated (append-only is preserved).
    # ------------------------------------------------------------------

    @staticmethod
    def split_version_suffix(key: str) -> tuple[str, Optional[int]]:
        """'notes_v3' -> ('notes', 3); 'notes' -> ('notes', None)."""
        m = _VERSION_SUFFIX_RE.match(key)
        if m:
            return m.group("base"), int(m.group("num"))
        return key, None

    def family_versions(self, base: str) -> Dict[int, str]:
        """Map version number -> actual canvas key for a doc family.

        The plain base key, when present, is version 1. Returns {} if
        nothing in the family exists.
        """
        fam: Dict[int, str] = {}
        if base in self.canvas:
            fam[1] = base
        for key in self.canvas.keys():
            b, num = self.split_version_suffix(key)
            if b == base and num is not None:
                fam[num] = key
        return fam

    def _family_history_str(self, fam: Dict[int, str]) -> str:
        parts = [fam[num] for num in sorted(fam)]
        return ", ".join(parts)

    def inspect(self):
        """List every canvas key literally, including each version of a
        version-controlled doc (e.g. 'notes', 'notes_v2', 'notes_v3').
        No collapsing — the agent sees the full key set and picks the
        version it wants."""
        return list(self.canvas.keys())

    def read(self, key):
        if len(self.canvas) == 0:
            return "Canvas is empty"

        if key == "finished_job_list":
            # turn list into a dict where the key is the index and the value is the job name
            returning_finished_job_list = {i: job for i, job in enumerate(self.canvas[key].items())}
            return repr(returning_finished_job_list)

        # Literal lookup: 'xxx' returns exactly 'xxx'. To read the latest
        # version of a versioned doc, the agent specifies that exact
        # version key (visible in inspect()).
        if key in self.canvas:
            return f"{self.canvas[key]}"

        # Not found: if the key looks like it belongs to a doc family,
        # surface that family's existing keys so a wrong version guess is
        # self-correcting.
        base, _ = self.split_version_suffix(key)
        fam = self.family_versions(base)
        if fam:
            _dbg(f"[CANVAS read] key={key!r}: MISS; family '{base}' has "
                 f"{len(fam)} version(s) -> returning family hint.")
            return (f"Key '{key}' not found. Existing versions of doc "
                    f"'{base}': {self._family_history_str(fam)}")
        _dbg(f"[CANVAS read] key={key!r}: MISS (no such key/family).")
        return f"Key '{key}' not found. Please choose from {list(self.canvas.keys())}"
        
    def write(self, key, value, overwrite=False):
        writeDir = os.path.join(self.working_directory, 'canvas.pickle')
        # if key not in self.canvas:

        # ---- Guard 1: reserved framework prefixes -----------------------
        for prefix in RESERVED_KEY_PREFIXES:
            if key.startswith(prefix):
                _dbg(f"[CANVAS write] REJECTED key={key!r}: reserved prefix "
                     f"{prefix!r}.")
                return (f"Key '{key}' is rejected: the prefix '{prefix}' is "
                        f"reserved for framework-generated content. Please "
                        f"choose a different key.")

        # ---- Guard 2: version suffix is framework-owned ------------------
        _base, _num = self.split_version_suffix(key)
        if _num is not None:
            _dbg(f"[CANVAS write] REJECTED key={key!r}: '_v<N>' suffix is "
                 f"reserved for the version system (base={_base!r}).")
            return (f"Key '{key}' is rejected: keys ending in '_v<N>' are "
                    f"reserved for the version-controlled doc system. To "
                    f"create a doc, write it under the plain name "
                    f"'{_base}'; to revise an existing doc, use the "
                    f"version_controlled_edit_canvas_doc tool, which will "
                    f"create the next '_v<N>' version for you.")

        # ---- Guard 3: versioned families are append-only via edit_doc ----
        if overwrite and len(self.family_versions(key)) > 1:
            fam = self.family_versions(key)
            _dbg(f"[CANVAS write] REJECTED overwrite of key={key!r}: belongs "
                 f"to a versioned family ({self._family_history_str(fam)}).")
            return (f"Key '{key}' belongs to a version-controlled doc family "
                    f"({self._family_history_str(fam)}) and cannot be "
                    f"overwritten — that would silently rewrite version "
                    f"history. Use the version_controlled_edit_canvas_doc "
                    f"tool to create the next version instead.")

        if key in self.SpecialKeys:
            if key == "finished_job_list":
                return f"Key '{key}' is read-only and cannot be overwritten."
            assert isinstance(value, dict), f"Value for key '{key}' must be a dict."
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()), f"All keys and values in the dict for key '{key}' must be strings."
            # assert all(isinstance(t, tuple) and len(t) == 2 and all(isinstance(s, str) for s in t) for t in value), f"All elements in the list for key '{key}' must be tuples of (str(job_name), str(id))."
            
            
        if key not in self.canvas.keys():
            self.canvas[key] = value
            with open(writeDir, 'wb') as f:
                pickle.dump((self.canvas, self.result_registry), f)
            print("##################### CANVAS #######################")
            myDictPP(self.canvas, toDisk=True, filename=writeDir+'.txt')
            print("################### CANVAS END #####################")    
            _dbg(f"[CANVAS write] key={key!r} ADDED (canvas now has "
                 f"{len(self.canvas)} key(s)).")
            return f"Key '{key}' successfully added."
        elif key in var.tmp_report_names:
            _dbg(f"[CANVAS write] REJECTED key={key!r}: it is a structured "
                 f"report name (reports are immutable).")
            return f"Key '{key}' already exists and is one of the structured report names. You may never overwirte a report. Please choose a different key."
        elif overwrite:
            self.canvas[key] = value
            with open(writeDir, 'wb') as f:
                pickle.dump((self.canvas, self.result_registry), f)
            print("##################### CANVAS #######################")
            myDictPP(self.canvas, toDisk=True, filename=writeDir+'.txt')
            print("################### CANVAS END #####################")   
            _dbg(f"[CANVAS write] key={key!r} OVERWRITTEN.")
            return f"Key '{key}' successfully overwritten."
        else:
            _dbg(f"[CANVAS write] key={key!r} exists and overwrite=False -> "
                 f"refused.")
            return f"Key '{key}' already exists. Please choose a different key. If you want to overwrite the value, set the 'overwrite' flag to True."

    def _persist(self):
        writeDir = os.path.join(self.working_directory, 'canvas.pickle')
        with open(writeDir, 'wb') as f:
            pickle.dump((self.canvas, self.result_registry), f)
        print("##################### CANVAS #######################")
        myDictPP(self.canvas, toDisk=True, filename=writeDir + '.txt')
        print("################### CANVAS END #####################")

    def framework_write(self, key, value):
        """Framework-internal write that bypasses agent-facing guards.

        Used for reserved-prefix keys (JUDGE::, ARCHIVE::) and for
        version-suffixed keys created by edit_doc. Never exposed as an
        agent tool. Still refuses to mutate existing keys, preserving
        the append-only property.
        """
        if key in self.canvas:
            _dbg(f"[CANVAS framework_write] key={key!r} already exists "
                 f"(append-only) -> refused.")
            return False, f"framework_write: key '{key}' already exists (canvas is append-only)."
        self.canvas[key] = value
        self._persist()
        _dbg(f"[CANVAS framework_write] key={key!r} added.")
        return True, f"Key '{key}' successfully added."

    def framework_write_versioned(self, base: str, value):
        """Framework-internal write that appends the NEXT free version of a
        doc family without ever mutating an existing key.

        Writes the plain `base` if the family is empty, else `base_v{max+1}`.
        Returns (actual_key, message). Used for framework-owned records that
        can legitimately recur for the same logical subject (e.g. a report
        re-judged after resubmission -> JUDGE::<name>, JUDGE::<name>_v2; a
        re-compressed step span -> ARCHIVE::...). `base` must NOT itself end
        in a version suffix.
        """
        b, num = self.split_version_suffix(base)
        if num is not None:
            base = b  # normalize: never produce nested suffixes
        fam = self.family_versions(base)
        if not fam:
            actual_key = base
        else:
            actual_key = f"{base}_v{max(fam) + 1}"
        self.canvas[actual_key] = value
        self._persist()
        _dbg(f"[CANVAS framework_write_versioned] base={base!r} -> "
             f"actual_key={actual_key!r} (family had {len(fam)} prior "
             f"version(s)).")
        return actual_key, f"Key '{actual_key}' successfully added."

    def edit_doc(self, name: str, edits: List[Dict[str, str]]):
        """Apply edits to the LATEST version of a doc family and store the
        result as the next '_v<N>' key. Existing versions are never
        mutated (append-only).

        `name` may be the plain base name, or the latest version's full
        key — any trailing '_v<N>' is stripped to find the family, so
        nested suffixes like 'notes_v2_v3' can never be produced.
        Referencing a STALE version explicitly is an error: the caller
        composed its old_str against content it last read, and edits
        only ever apply to the latest version.

        Each edit is a dict:
          {"mode": "replace", "old_str": <unique snippet>, "new_str": ...}
          {"mode": "append",  "new_str": <text to append>}
        Edits apply sequentially and atomically: if any edit fails, no
        new version is written.
        """
        base, requested_num = self.split_version_suffix(name)
        fam = self.family_versions(base)
        _dbg(f"[CANVAS edit_doc] name={name!r} (base={base!r}) with "
             f"{len(edits)} edit op(s); family has {len(fam)} version(s).")
        if not fam:
            _dbg(f"[CANVAS edit_doc] FAILED: no doc named '{base}'.")
            return (f"EDIT_FAILED: no doc named '{base}' exists on the "
                    f"canvas. Create it first with write_my_canvas under "
                    f"the plain name '{base}' (no '_v<N>' suffix).")

        latest_num = max(fam)
        latest_key = fam[latest_num]

        if requested_num is not None and requested_num != latest_num:
            _dbg(f"[CANVAS edit_doc] FAILED: stale version {name!r} "
                 f"(latest is {latest_key!r}).")
            return (f"EDIT_FAILED: you referenced version '{name}', but the "
                    f"latest version of this doc is '{latest_key}' "
                    f"(v{latest_num}). Edits only apply to the latest "
                    f"version. Read '{latest_key}' first, then re-issue "
                    f"your edits against its content (you can pass the "
                    f"plain name '{base}').")

        content = self.canvas[latest_key]
        if latest_key in var.tmp_report_names or base in var.tmp_report_names:
            _dbg(f"[CANVAS edit_doc] FAILED: '{latest_key}' is an immutable "
                 f"structured report.")
            return (f"EDIT_FAILED: '{latest_key}' is a structured "
                    f"scientific report. Reports are immutable — if the "
                    f"report is wrong, fix the upstream data and generate "
                    f"a new report via generate_structured_report.")
        if not isinstance(content, str):
            return (f"EDIT_FAILED: doc '{latest_key}' holds a "
                    f"{type(content).__name__}, not text. The edit tool "
                    f"only works on string content.")

        new_content = content
        for i, op in enumerate(edits):
            mode = op.get("mode")
            old_str = op.get("old_str", "") or ""
            new_str = op.get("new_str", "")
            if new_str is None:
                new_str = ""
            if mode == "replace":
                if not old_str:
                    return (f"EDIT_FAILED at edit #{i+1}: mode='replace' "
                            f"requires a non-empty old_str. No new version "
                            f"was written.")
                n = new_content.count(old_str)
                if n == 0:
                    return (f"EDIT_FAILED at edit #{i+1}: old_str was not "
                            f"found in the latest version '{latest_key}'. "
                            f"The content may have changed since you last "
                            f"read it (or a previous edit in this same call "
                            f"already altered it) — read '{base}' again and "
                            f"retry. No new version was written.")
                if n > 1:
                    return (f"EDIT_FAILED at edit #{i+1}: old_str appears "
                            f"{n} times in '{latest_key}' and must be "
                            f"unique. Extend the snippet with surrounding "
                            f"text to disambiguate. No new version was "
                            f"written.")
                new_content = new_content.replace(old_str, new_str, 1)
            elif mode == "append":
                if not new_str:
                    return (f"EDIT_FAILED at edit #{i+1}: mode='append' "
                            f"requires a non-empty new_str. No new version "
                            f"was written.")
                sep = "" if (new_content.endswith("\n") or not new_content) else "\n"
                new_content = new_content + sep + new_str
            else:
                return (f"EDIT_FAILED at edit #{i+1}: unknown mode "
                        f"{mode!r}. Use 'replace' or 'append'. No new "
                        f"version was written.")

        if new_content == content:
            return (f"EDIT_FAILED: the edits produced content identical to "
                    f"'{latest_key}'. No new version was written.")

        new_key = f"{base}_v{latest_num + 1}"
        self.canvas[new_key] = new_content
        self._persist()
        _dbg(f"[CANVAS edit_doc] SUCCESS: '{latest_key}' -> '{new_key}' "
             f"({len(edits)} edit(s) applied; {len(content)} -> "
             f"{len(new_content)} chars).")
        return (f"Doc '{base}' updated: new version '{new_key}' created "
                f"from '{latest_key}' ({len(edits)} edit(s) applied). "
                f"To read the latest content, read '{new_key}'. Previous "
                f"versions remain available under their own keys for "
                f"history.")

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
        context: str = "",
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
        writeDir = os.path.join(self.working_directory, 'canvas.pickle')
        try:
            value = float(value)
            artifact = NumericArtifact(
                result_id=result_id,
                tool_name=tool_name,
                args=args,
                value=value,
                description=description,
                context=context,
                reasons=reasons,
                parent_result_ids=parent_result_ids or [],
                parent_result_ids_w_args=parent_result_ids_w_args or {},
                metadata=metadata or {},
            )
            # add duplication check
            self.result_registry[result_id] = artifact
            self.curr_round_result_ids.append(result_id)
            
            with open(writeDir, 'wb') as f:
                pickle.dump((self.canvas, self.result_registry), f)
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
                            context=context,
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
                            context=context,
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
                    context=context,
                    parent_result_ids=parent_result_ids or [],
                    parent_result_ids_w_args=parent_result_ids_w_args or {},
                    metadata=metadata or {},
                )
                self.result_registry[result_id] = artifact
                self.curr_round_result_ids.append(result_id)
                with open(writeDir, 'wb') as f:
                    pickle.dump((self.canvas, self.result_registry), f)
                return result_id
            else:
                artifact = OtherArtifact(
                    result_id=result_id,
                    tool_name=tool_name,
                    args=args,
                    value=value,
                    description=description,
                    context=context,
                    reasons=reasons,
                    parent_result_ids=parent_result_ids or [],
                    parent_result_ids_w_args=parent_result_ids_w_args or {},
                    metadata=metadata or {},
                )
                self.result_registry[result_id] = artifact
                self.curr_round_result_ids.append(result_id)
                with open(writeDir, 'wb') as f:
                    pickle.dump((self.canvas, self.result_registry), f)
                return result_id
        
    def get_artifact(self, result_id: str):
        return self.result_registry.get(result_id, None)

    def _compare_to(self, expected_value: Any, actual_value: Any, tol: float):
        """Type-aware value comparison used by verify_artifact for both
        bare-ref and dotted-ref code paths. Returns (ok, mismatch_msg).
        mismatch_msg is the empty string on success."""
        if isinstance(actual_value, (int, float)):
            try:
                ev = float(expected_value)
                if math.isclose(ev, float(actual_value), rel_tol=0.0, abs_tol=tol):
                    return True, ""
                return False, (
                    f"Expected value: {repr(expected_value)} does not "
                    f"match registered value {repr(actual_value)}."
                )
            except (ValueError, TypeError):
                return False, (
                    f"Expected value: {repr(expected_value)} cannot be "
                    f"compared numerically to {repr(actual_value)}."
                )
        if isinstance(actual_value, np.ndarray):
            try:
                ev = np.array(expected_value)
                if np.all(np.isclose(actual_value, ev, rtol=0.0, atol=tol)):
                    return True, ""
                return False, (
                    f"Expected value: {repr(expected_value)} does not "
                    f"match registered value {repr(actual_value)}."
                )
            except Exception:
                return False, (
                    f"Expected value: {repr(expected_value)} cannot be "
                    f"compared elementwise to {repr(actual_value)}."
                )
        # Fallback: exact equality.
        if expected_value == actual_value:
            return True, ""
        return False, (
            f"Expected value: {repr(expected_value)} does not match "
            f"registered value {repr(actual_value)}."
        )

    def verify_artifact(
        self,
        expected_value: Any,
        source_result_id,
        tol: float = 1e-7,
    ) -> tuple[bool, str, Optional[NumericArtifact]]:
        if not isinstance(source_result_id, str) or not source_result_id:
            return False, "Invalid ref ID: must be a non-empty string."

        # ---- Dotted-ref path: '<8-char-id>.<args-key>' ----
        # Use this to reference a SPECIFIC INPUT PARAMETER of a past tool
        # call, rather than the call's output. Only top-level args keys
        # are addressable; nested dict/list fields are not.
        if "." in source_result_id:
            parts = source_result_id.split(".")
            if len(parts) != 2:
                return False, (
                    f"Malformed dotted ref '{source_result_id}'. Expected "
                    f"'<8-char-id>.<input-param-name>' — exactly one dot, "
                    f"with the bare id on the left and a single args key "
                    f"on the right. Nested args fields are not "
                    f"addressable."
                )
            bare_id, field = parts
            if len(bare_id) != 8:
                return False, (
                    f"Malformed dotted ref '{source_result_id}': the part "
                    f"before the dot must be an 8-character result_id."
                )
            if not field:
                return False, (
                    f"Malformed dotted ref '{source_result_id}': the part "
                    f"after the dot (the input-param name) is empty."
                )

            artifact = self.result_registry.get(bare_id)
            if artifact is None:
                return False, (
                    f"ID {bare_id} (from dotted ref '{source_result_id}') "
                    f"does not exist in CANVAS. Confirm the bare id is "
                    f"correct."
                )

            args = artifact.args or {}
            if field not in args:
                arg_names = list(args.keys())
                if not arg_names:
                    arg_list_str = "<no args recorded on this artifact>"
                elif len(arg_names) == 1:
                    arg_list_str = repr(arg_names[0])
                elif len(arg_names) == 2:
                    arg_list_str = f"{arg_names[0]!r} and {arg_names[1]!r}"
                else:
                    arg_list_str = (
                        ", ".join(repr(n) for n in arg_names[:-1])
                        + f", and {arg_names[-1]!r}"
                    )
                return False, (
                    f"Dotted ref '{source_result_id}': artifact "
                    f"'{bare_id}' has no input parameter named '{field}'. "
                    f"Available args are {arg_list_str}."
                )

            actual = args[field]
            ok, msg = self._compare_to(expected_value, actual, tol)
            if ok:
                return True, (
                    f"{source_result_id} Verification success (input "
                    f"parameter '{field}' of artifact '{bare_id}')."
                )
            return False, (
                f"Dotted ref '{source_result_id}' verification failed. "
                f"{msg}"
            )

        # ---- Bare-ref path: existing behavior, unchanged in semantics. ----
        if len(source_result_id) != 8:
            return False, f"Invalid ref ID format: Tool generated ID would be an 8-character string. Please check CANVAS and find the correct ID to reference."

        artifact = self.result_registry.get(source_result_id)
        if artifact is None:
            return False, f"ID {source_result_id} does not exist. Please check CANVAS and find the correct ID to reference."

        if isinstance(artifact, ListedArtifact):
            # As long as one of the artifacts in the list matches the
            # expected value, we consider it a success.
            for sub_artifact in artifact.value:
                ok, _ = self._compare_to(expected_value, sub_artifact.value, tol)
                if ok:
                    return True, f"{source_result_id} Verification success."
            return False, (
                f"ID {source_result_id} listed verification failed. "
                f"\nExpected value: {repr(expected_value)} does not match "
                f"any of the registered tool outputs "
                f"{[repr(sub_artifact.value) for sub_artifact in artifact.value]}."
            )

        ok, msg = self._compare_to(expected_value, artifact.value, tol)
        if ok:
            return True, f"{source_result_id} Verification success."
        return False, f"ID {source_result_id} verification failed. \n{msg}"

        
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