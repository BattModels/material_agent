"""
live_visualizer.py
==================

Drop-in on-the-fly HTML visualizer for a LangGraph agentic run with a
supervisor / worker / boss topology (the same shape you use in your OER
screening pipeline).

USAGE
-----

    from live_visualizer import LiveVisualizer

    # Create once, near the top of your run:
    viz = LiveVisualizer(
        working_directory=var.my_WORKING_DIRECTORY,
        canvas_obj=CANVAS,       # has a `.canvas` dict
        explog_obj=EXPLOG,       # has `.relational_frame.<name>.df`
    )

    # Then, inside your existing print_stream, add ONE line at the top:
    def print_stream(s, DAG=None):
        viz.on_event(s, DAG=DAG)      # <-- add this line
        # ... your existing code unchanged ...

    # Optional: when the run finishes
    viz.close()

OUTPUT
------
In `working_directory` you get:

    live_visualization.html   (written once — open this in a browser)
    live_data.js              (rewritten on every event, polled by the HTML)

The HTML polls `live_data.js` via a cache-busted `<script>` tag every
second, so it works from a plain `file://` URL — no local web server
needed (though `python -m http.server` works too).

PARSING MODEL
-------------
The class hooks into every chunk yielded by `graph.stream()`.  A chunk is
either a state dict (no `"messages"` key) or a message chunk.  It detects
agent boundaries from two signals:

  1. The outer key of a node-output update: `{'Supervisor': {...}}`,
     `{'OER_Agent': {...}}`, etc.  This tells it who just finished.
  2. Structured-output tool messages: `Act` (supervisor), `wokerResponse`
     (worker), `BossReview` (boss).  When one of these arrives, the
     current step is closed and a new one is started below it.

For the supervisor's `Act` response it distinguishes `Plan`, `NoChange`,
and `Response` variants and renders a readable summary for each.
"""

import os
import json
import re
import time
from datetime import datetime
from threading import Lock

from src import var

try:
    import pandas as pd
except ImportError:  # pandas is required only for DataFrame handling
    pd = None


# --------------------------------------------------------------------------
# Tool names used for structured output in your framework
# --------------------------------------------------------------------------
SUPERVISOR_TOOL = "Act"
WORKER_TOOL = "wokerResponse"     # (sic — matches your class name)
BOSS_TOOL = "BossReview"
STRUCTURED_TOOLS = {SUPERVISOR_TOOL, WORKER_TOOL, BOSS_TOOL}


# --------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------
def _safe_repr(obj, max_len=60000):
    """Get a repr of anything, with DataFrames rendered nicely, truncated."""
    try:
        if pd is not None and isinstance(obj, pd.DataFrame):
            if obj.empty:
                return f"<empty DataFrame, columns={list(obj.columns)}>"
            s = obj.to_string(max_rows=80, max_cols=30)
        elif isinstance(obj, str):
            s = obj
        else:
            s = repr(obj)
    except Exception as e:
        s = f"<unreprable: {e}>"
    if len(s) > max_len:
        return s[:max_len] + f"\n... [truncated, total {len(s)} chars]"
    return s


def _is_simple(v):
    """True if v is a scalar / short string that belongs on one line."""
    return isinstance(v, (bool, int, float, type(None))) or (
        isinstance(v, str) and len(v) < 300 and "\n" not in v
    )


def _dict_to_markdown(d, depth=0, max_depth=6):
    """Convert a (possibly nested) dict into readable markdown.

    depth 0 → sub-dicts get ``## key`` headings
    depth 1 → ``### key``
    depth 2+ → ``**key:**`` with indented content
    Lists   → bullet points
    Scalars → ``**key:** value``
    DataFrames, unknown objects → ``_safe_repr``
    """
    if depth > max_depth:
        return _safe_repr(d, 4000)

    lines = []
    for key, val in (d.items() if isinstance(d, dict) else [(None, d)]):
        label = str(key) if key is not None else ""

        # --- dict value → recurse as a section ---
        if isinstance(val, dict):
            if depth == 0:
                lines.append(f"\n## {label}\n")
            elif depth == 1:
                lines.append(f"\n### {label}\n")
            else:
                lines.append(f"\n**{label}:**\n")
            lines.append(_dict_to_markdown(val, depth + 1, max_depth))

        # --- list value → bullet points ---
        elif isinstance(val, (list, tuple)):
            lines.append(f"**{label}:**" if label else "")
            for item in val:
                if isinstance(item, dict):
                    # nested dict inside a list: render inline
                    inner = _dict_to_markdown(item, depth + 2, max_depth)
                    # indent each line of the inner block under the bullet
                    indented = inner.strip().replace("\n", "\n  ")
                    lines.append(f"- {indented}")
                else:
                    lines.append(f"- {_format_scalar(item)}")

        # --- simple scalar → one-liner ---
        elif _is_simple(val):
            if label:
                lines.append(f"**{label}:** {_format_scalar(val)}")
            else:
                lines.append(_format_scalar(val))

        # --- long string (may already be markdown) → block ---
        elif isinstance(val, str):
            if label:
                lines.append(f"**{label}:**")
            lines.append(val)

        # --- DataFrame ---
        elif pd is not None and isinstance(val, pd.DataFrame):
            if label:
                lines.append(f"**{label}:** (DataFrame, {val.shape[0]} rows × {val.shape[1]} cols)")
            lines.append(f"```\n{_safe_repr(val, 8000)}\n```")

        # --- anything else → repr ---
        else:
            if label:
                lines.append(f"**{label}:** `{_safe_repr(val, 500)}`")
            else:
                lines.append(f"`{_safe_repr(val, 500)}`")

    return "\n".join(lines)


def _format_scalar(v):
    """Render a scalar value for inline display."""
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        # Avoid ugly floats like 1.8000000000000004
        s = f"{v:.6g}"
        return s
    return str(v)


def _df_to_records(df):
    """Convert a DataFrame to list-of-dicts for JSON serialization."""
    if pd is None or df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col, val in row.items():
            if val is None:
                rec[str(col)] = None
                continue
            try:
                if pd.isna(val):
                    rec[str(col)] = None
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(val, (bool, int, float, str)):
                rec[str(col)] = val
            else:
                rec[str(col)] = _safe_repr(val, 500)
        records.append(rec)
    return records


def _truncate(s, max_len):
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    if len(s) > max_len:
        return s[:max_len] + "…"
    return s


# --------------------------------------------------------------------------
# LiveVisualizer
# --------------------------------------------------------------------------
class LiveVisualizer:

    def __init__(
        self,
        canvas_obj=None,
        explog_obj=None,
        html_filename="live_visualization.html",
        data_filename="live_data.js",
        poll_interval_ms=1000,
        title="DREAMS material screening",
        known_agents=("Supervisor", "Boss"),
        dag_filename_pattern="step_{id}_DAG.html",
        hide_columns=None,
    ):
        self.html_filename = html_filename
        self.data_filename = data_filename
        self.canvas_obj = canvas_obj
        self.explog_obj = explog_obj
        self.title = title
        self.known_agents = set(known_agents)
        self.poll_interval_ms = poll_interval_ms
        self.dag_filename_pattern = dag_filename_pattern
        # Columns to hide from each EXPLOG tab. Configurable; defaults match
        # your request (candidates: study_obj; processes: VASP_dir).
        default_hide = {
            "candidates": {"study_obj"},
            "processes": {"VASP_dir"},
        }
        if hide_columns:
            for k, v in hide_columns.items():
                default_hide.setdefault(k, set()).update(set(v))
        self.hide_columns = {k: set(v) for k, v in default_hide.items()}
        self._lock = Lock()
        # Write-throttle state (see var.LIVE_VIZ_*). Both output files are views
        # rebuilt from in-memory state, so a skipped write loses nothing.
        self._last_data_flush = 0.0
        self._last_html_flush = 0.0
        self._skipped_flushes = 0

        self.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.steps = []
        self.next_step_id = 1
        self.current_step = self._new_step()
        # when an AIMessage contains a structured-output tool call, we stash
        # its clean dict here so the matching ToolMessage can use it
        self.pending_structured_args = None

        self.tokens_total = {"input": 0, "output": 0}
        self.current_agent_guess = "Supervisor"
        self.pending_inputs_text = None
        self.dag_links = {}                # step_id -> filename (for step-card badges)
        self.latest_dag_filename = None    # most recently written DAG file
        self.latest_dag_mtime = None       # its mtime, for iframe cache-busting
        self._dag_content_cache = None     # cached DAG HTML content
        self._dag_content_mtime = None     # mtime when we last read the file


    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def set_working_directory(self, working_directory):
        self.working_directory = working_directory
        os.makedirs(working_directory, exist_ok=True)
        self.html_path = os.path.join(working_directory, self.html_filename)
        self.data_path = os.path.join(working_directory, self.data_filename)
        
        # --- RESUME: restore state from a previous run if data file exists ---
        self._try_restore_state()

        # Pre-split the HTML template at the data placeholder so we can
        # efficiently inject embedded data on every flush.
        rendered_tpl = HTML_TEMPLATE.replace("__DATA_FILE__", os.path.basename(self.data_path))
        rendered_tpl = rendered_tpl.replace("__POLL_MS__", str(self.poll_interval_ms))
        rendered_tpl = rendered_tpl.replace("__TITLE__", self.title)
        marker = "/* __EMBEDDED_DATA_JSON__ */null"
        idx = rendered_tpl.index(marker)
        self._html_prefix = rendered_tpl[:idx]
        self._html_suffix = rendered_tpl[idx + len(marker):]

        self._flush(force=True)   # first paint
    
    def on_event(self, s, DAG=None):
        """Call this for every chunk yielded by graph.stream()."""
        with self._lock:
            try:
                if DAG is not None:
                    # remember that this step (the most recently closed one,
                    # or the current one if none closed yet) has a DAG html
                    fname = self.dag_filename_pattern.format(id=DAG)
                    target_id = self.steps[-1]["id"] if self.steps else self.current_step["id"]
                    self.dag_links[target_id] = fname
                    # Track it as the "latest DAG" so the iframe in the DAG
                    # tab can cache-bust via mtime whenever gen_DAG rewrites
                    # the file.
                    self.latest_dag_filename = fname
                    try:
                        self.latest_dag_mtime = os.path.getmtime(
                            os.path.join(self.working_directory, fname)
                        )
                    except OSError:
                        self.latest_dag_mtime = None

                if isinstance(s, dict) and "messages" not in s:
                    self._handle_state(s)
                elif isinstance(s, dict) and "messages" in s:
                    msgs = s["messages"]
                    if msgs:
                        self._handle_message(msgs[-1])
                # other shapes silently ignored
                self._flush()
            except Exception as e:
                # never break the agent run on a viz error
                import traceback
                print(f"[LiveVisualizer] error handling event: {e}")
                traceback.print_exc()

    def close(self):
        """Mark end of session and flush once more."""
        with self._lock:
            # Only save the current step if it has actual content (events).
            # A step that was merely pre-tagged with an agent name (via `next`
            # from a node output) is just a placeholder and shouldn't be saved.
            if self.current_step and self.current_step.get("events"):
                self.current_step["status"] = "completed"
                if not self.current_step.get("title"):
                    self.current_step["title"] = f"{self.current_step.get('agent') or 'Session'} — end"
                self.current_step["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.steps.append(self.current_step)
                self.current_step = self._new_step()
            self._flush(force=True)   # final state must always land

    # ------------------------------------------------------------------
    # STATE HANDLING
    # ------------------------------------------------------------------
    def _handle_state(self, s):
        # node output update: single top-level key matching a node name
        if len(s) == 1:
            key = next(iter(s))
            inner = s[key]
            if isinstance(inner, dict) and self._looks_like_node_name(key):
                self._on_node_output(key, inner)
                return

        # input / values snapshot — has multiple top-level keys
        if isinstance(s, dict) and "inputs" in s and self.pending_inputs_text is None:
            txt = s.get("inputs", "")
            self.pending_inputs_text = _truncate(str(txt), 3000)
            if self.current_step and not self.current_step.get("detail"):
                self.current_step["detail"] = (
                    "**Session inputs:**\n" + _truncate(self.pending_inputs_text, 2000)
                )

    def _looks_like_node_name(self, key):
        if key in self.known_agents:
            return True
        # heuristic: CamelCase or snake_case identifier with _Agent / Agent in it
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            if key.endswith("Agent") or key.endswith("_Agent"):
                self.known_agents.add(key)
                return True
            # Supervisor / Boss / anything explicitly named
            if key.lower() in {"supervisor", "boss"}:
                self.known_agents.add(key)
                return True
        return False

    def _on_node_output(self, node_name, updates):
        """A node just finished. Attribute the streaming step (if any) to it."""
        self.current_agent_guess = node_name
        # If the current step has any content but no agent yet, tag it
        if self.current_step.get("events") and not self.current_step.get("agent"):
            self.current_step["agent"] = node_name

        # If the streaming step is non-empty but has no structured-response
        # close (shouldn't happen in the normal flow, but guard anyway), close it
        if (
            self.current_step.get("events")
            and not self.current_step.get("structured_response")
        ):
            self.current_step["agent"] = self.current_step.get("agent") or node_name
            self.current_step["status"] = "completed"
            self.current_step["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if not self.current_step.get("title"):
                self.current_step["title"] = self._auto_title(node_name, self.current_step)
            if not self.current_step.get("summary"):
                self.current_step["summary"] = "(node returned without structured response)"
            self.steps.append(self.current_step)
            self.current_step = self._new_step()
            self.pending_structured_args = None

        # If updates carries a `next` field, pre-tag the next step's agent
        nxt = updates.get("next")
        if nxt and isinstance(nxt, str):
            self.current_step["agent"] = nxt

    # ------------------------------------------------------------------
    # MESSAGE HANDLING
    # ------------------------------------------------------------------
    def _handle_message(self, msg):
        # skip tuples (e.g. (AIMessageChunk, metadata) from messages stream mode)
        if isinstance(msg, tuple):
            return

        msg_type = type(msg).__name__

        # token accounting (usage_metadata is only present on AIMessage)
        usage = getattr(msg, "usage_metadata", None)
        if isinstance(usage, dict):
            self.tokens_total["input"] += int(usage.get("input_tokens") or 0)
            self.tokens_total["output"] += int(usage.get("output_tokens") or 0)

        if msg_type in ("AIMessage", "AIMessageChunk"):
            self._handle_ai_message(msg)
        elif msg_type == "ToolMessage":
            self._handle_tool_message(msg)
        elif msg_type == "HumanMessage":
            content = getattr(msg, "content", "")
            if content and self.current_step:
                self.current_step["events"].append({
                    "type": "human_input",
                    "content": _truncate(str(content), 4000),
                    "time": self._now_time(),
                })
        else:
            content = getattr(msg, "content", str(msg))
            if self.current_step:
                self.current_step["events"].append({
                    "type": "other",
                    "msg_type": msg_type,
                    "content": _truncate(str(content), 4000),
                    "time": self._now_time(),
                })

    def _handle_ai_message(self, msg):
        tool_calls = getattr(msg, "tool_calls", None) or []
        content = getattr(msg, "content", "")

        if tool_calls:
            for tc in tool_calls:
                name, args, tc_id = self._parse_tool_call(tc)
                if name in STRUCTURED_TOOLS:
                    # stash the clean dict args for the matching ToolMessage
                    self.pending_structured_args = (name, args, tc_id)
                    self.current_step["events"].append({
                        "type": "structured_call",
                        "name": name,
                        "args_preview": _truncate(
                            json.dumps(args, default=str, ensure_ascii=False), 400
                        ),
                        "time": self._now_time(),
                    })
                else:
                    self.current_step["events"].append({
                        "type": "tool_call",
                        "name": name,
                        "args_preview": _truncate(
                            json.dumps(args, default=str, ensure_ascii=False), 240
                        ),
                        "args_full": _safe_repr(args, 8000),
                        "id": tc_id,
                        "result": None,
                        "result_full": None,
                        "time": self._now_time(),
                    })
            return

        # plain AI text (rare in this framework, but some nodes stream it)
        if content:
            if isinstance(content, list):
                # Anthropic sometimes returns list-of-blocks content
                content = " ".join(
                    (c.get("text", "") if isinstance(c, dict) else str(c))
                    for c in content
                )
            self.current_step["events"].append({
                "type": "ai_text",
                "content": _truncate(str(content), 8000),
                "time": self._now_time(),
            })

    def _parse_tool_call(self, tc):
        """Normalize a LangChain tool call to (name, args_dict, id_str)."""
        if isinstance(tc, dict):
            name = tc.get("name", "")
            args = tc.get("args", {})
            tc_id = tc.get("id", "")
        else:  # ToolCall object
            name = getattr(tc, "name", "")
            args = getattr(tc, "args", {})
            tc_id = getattr(tc, "id", "")
        if not isinstance(args, dict):
            args = {"value": args}
        return name, args, tc_id

    def _handle_tool_message(self, msg):
        name = getattr(msg, "name", "") or ""
        content = str(getattr(msg, "content", "") or "")
        tc_id = getattr(msg, "tool_call_id", "") or ""

        if name in STRUCTURED_TOOLS:
            self._finalize_current_step(name, content, self.pending_structured_args)
            self.pending_structured_args = None
            return

        # regular tool result — try to attach it to its pending tool_call
        matched = False
        for ev in reversed(self.current_step["events"]):
            if ev.get("type") == "tool_call" and ev.get("result") is None:
                if (tc_id and ev.get("id") == tc_id) or (not tc_id and ev.get("name") == name):
                    ev["result"] = _truncate(content, 400)
                    ev["result_full"] = _truncate(content, 20000)
                    matched = True
                    break

        if not matched:
            self.current_step["events"].append({
                "type": "tool_result",
                "name": name,
                "content": _truncate(content, 400),
                "content_full": _truncate(content, 20000),
                "time": self._now_time(),
            })

    # ------------------------------------------------------------------
    # FINALIZE STEP ON STRUCTURED RESPONSE
    # ------------------------------------------------------------------
    def _finalize_current_step(self, structured_name, raw_content, pending):
        agent_guess = self.current_step.get("agent") or self.current_agent_guess
        if structured_name == SUPERVISOR_TOOL:
            agent_guess = "Supervisor"
        elif structured_name == BOSS_TOOL:
            agent_guess = "Boss"

        # get parsed payload from the stashed clean dict if available,
        # else regex-parse the repr string
        payload = pending[1] if (pending and len(pending) >= 2) else None
        parsed = self._parse_structured(structured_name, raw_content, payload)

        title, summary, tags, phase = self._derive_step_meta(
            agent_guess, structured_name, parsed
        )

        self.current_step["agent"] = agent_guess
        self.current_step["status"] = "completed"
        self.current_step["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.current_step["title"] = title
        self.current_step["summary"] = summary
        self.current_step["tags"] = tags
        self.current_step["phase"] = phase
        self.current_step["structured_response"] = {
            "type": structured_name,
            "parsed": parsed,
            "raw": _truncate(raw_content, 8000),
        }
        self.steps.append(self.current_step)
        self.current_step = self._new_step()

    def _parse_structured(self, name, content, payload):
        """Return a dict describing the structured response."""
        result = {}

        # Primary: use the clean dict from AIMessage.tool_calls[*].args
        if isinstance(payload, dict) and payload:
            if name == SUPERVISOR_TOOL:
                action = payload.get("action", payload)
                if isinstance(action, dict):
                    if "steps" in action:
                        result["action_type"] = "Plan"
                        result["steps"] = action.get("steps", [])
                    elif "response" in action:
                        result["action_type"] = "Response"
                        result["response"] = action.get("response", "")
                    elif "comment" in action:
                        result["action_type"] = "NoChange"
                        result["comment"] = action.get("comment", "")
                    else:
                        result["action_type"] = "Unknown"
                        result["raw"] = action
            elif name == WORKER_TOOL:
                result["answer"] = payload.get("answer", "")
                result["summary"] = payload.get("summary", "")
                result["success"] = bool(payload.get("success", True))
            elif name == BOSS_TOOL:
                result["decision"] = payload.get("decision", "")
                result["feedback"] = payload.get("feedback", "")
            if result:
                return result

        # Fallback: regex-parse the repr-style content
        if not content:
            return result

        if name == SUPERVISOR_TOOL:
            if "action=Plan(" in content:
                result["action_type"] = "Plan"
                step_texts = re.findall(r"step=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
                agent_texts = re.findall(r"agent=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
                steps = []
                for i, (_, st) in enumerate(step_texts):
                    ag = agent_texts[i][1] if i < len(agent_texts) else ""
                    steps.append({"step": st, "agent": ag})
                result["steps"] = steps
            elif "action=NoChange(" in content:
                result["action_type"] = "NoChange"
                m = re.search(r"comment=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
                result["comment"] = m.group(2) if m else ""
            elif "action=Response(" in content:
                result["action_type"] = "Response"
                m = re.search(r"response=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
                result["response"] = m.group(2) if m else ""
            elif "Error:" in content or "Failed" in content:
                result["action_type"] = "Error"
                result["error"] = _truncate(content, 1500)

        elif name == WORKER_TOOL:
            m = re.search(r"answer=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
            if m:
                result["answer"] = m.group(2)
            m = re.search(r"summary=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
            if m:
                result["summary"] = m.group(2)
            m = re.search(r"success=(True|False)", content)
            if m:
                result["success"] = (m.group(1) == "True")

        elif name == BOSS_TOOL:
            m = re.search(r"decision=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
            if m:
                result["decision"] = m.group(2)
            m = re.search(r"feedback=(['\"])((?:(?!\1).)*?)\1", content, re.DOTALL)
            if m:
                result["feedback"] = m.group(2)

        return result

    def _derive_step_meta(self, agent, structured_name, parsed):
        tags = []
        phase = "Execution"
        title = agent or "Step"
        summary = ""
        atype = parsed.get("action_type") if isinstance(parsed, dict) else None

        if structured_name == SUPERVISOR_TOOL:
            tags.append("supervisor")
            if atype == "Plan":
                tags.append("planning")
                phase = "Planning"
                steps = parsed.get("steps", []) or []
                title = f"Supervisor — Plan ({len(steps)} step{'s' if len(steps) != 1 else ''})"
                lines = []
                for s in steps[:10]:
                    if isinstance(s, dict):
                        st = _truncate(str(s.get("step", "")), 220)
                        ag = s.get("agent", "")
                        lines.append(f"→ {st}  [{ag}]")
                    else:
                        lines.append(f"→ {_truncate(str(s), 220)}")
                summary = "\n".join(lines)
                if len(steps) > 10:
                    summary += f"\n…and {len(steps) - 10} more"
            elif atype == "NoChange":
                tags.append("waiting")
                phase = "Continue"
                title = "Supervisor — No change"
                summary = _truncate(parsed.get("comment", "Continue executing the plan."), 400)
            elif atype == "Response":
                tags.append("results")
                phase = "Finalizing"
                title = "Supervisor — Draft final response"
                summary = _truncate(parsed.get("response", ""), 600)
            elif atype == "Error":
                tags.append("error")
                phase = "Error"
                title = "Supervisor — Structured-output parse error"
                summary = _truncate(parsed.get("error", ""), 600)
            else:
                title = "Supervisor"

        elif structured_name == WORKER_TOOL:
            tags.append("agent")
            if parsed.get("success") is False:
                tags.append("error")
                phase = "Failed"
                title = f"{agent} — ❌ Task failed"
            else:
                tags.append("execution")
                title = f"{agent} — Worker response"
            summary = _truncate(parsed.get("answer", ""), 600)

        elif structured_name == BOSS_TOOL:
            tags.append("supervisor")
            decision = (parsed.get("decision") or "").lower()
            if decision == "approve":
                tags.append("execution")
                phase = "Approved"
                title = "Boss — ✅ Approved"
                summary = "Draft response approved for delivery to user."
            else:
                tags.append("error")
                phase = "Revise"
                title = "Boss — ↩ Requested revision"
                summary = _truncate(parsed.get("feedback", ""), 600)

        return title, summary, tags, phase

    # ------------------------------------------------------------------
    # MISC
    # ------------------------------------------------------------------
    def _new_step(self):
        step = {
            "id": self.next_step_id,
            "agent": None,
            "phase": "",
            "status": "streaming",
            "start_time": self._now_full(),
            "end_time": None,
            "title": "",
            "summary": "",
            "detail": "",
            "tags": [],
            "events": [],
            "structured_response": None,
        }
        self.next_step_id += 1
        return step

    def _auto_title(self, agent, step):
        events = step.get("events", [])
        if not events:
            return agent or "Step"
        first = events[0]
        if first.get("type") == "tool_call":
            return f"{agent} — {first.get('name', 'tool')}"
        return f"{agent}"

    def _now_full(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _now_time(self):
        return datetime.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------------
    # COLLECT LIVE CANVAS / EXPLOG STATE
    # ------------------------------------------------------------------
    def _collect_canvas(self):
        if self.canvas_obj is None:
            return {}
        try:
            canvas_dict = getattr(self.canvas_obj, "canvas", None)
            if not isinstance(canvas_dict, dict):
                return {}
            out = {}
            for k, v in canvas_dict.items():
                if isinstance(v, dict):
                    out[str(k)] = _dict_to_markdown(v, depth=0)
                else:
                    out[str(k)] = _safe_repr(v, 80000)
            return out
        except Exception as e:
            return {"_error": f"Failed to read CANVAS: {e}"}

    def _collect_explog(self):
        out = {"candidates": [], "processes": []}
        if self.explog_obj is None:
            return out
        try:
            rf = getattr(self.explog_obj, "relational_frame", None)
            if rf is None:
                return out
            for cand_attr in ("candidates", "explog_candidates"):
                obj = getattr(rf, cand_attr, None)
                if obj is not None:
                    df = getattr(obj, "df", obj)
                    out["candidates"] = _df_to_records(df)
                    break
            for proc_attr in ("processes", "explog_processes"):
                obj = getattr(rf, proc_attr, None)
                if obj is not None:
                    df = getattr(obj, "df", obj)
                    out["processes"] = _df_to_records(df)
                    break
        except Exception as e:
            out["_error"] = f"Failed to read EXPLOG: {e}"
        return out

    # ------------------------------------------------------------------
    # FLUSH / WRITE
    # ------------------------------------------------------------------
    def _current_step_view(self):
        cs = self.current_step
        if not cs or (not cs.get("events") and not cs.get("agent")):
            return None
        # Cap very long event lists in live-streaming view so the JS payload
        # stays small (keeps the tail, which is what the user cares about).
        MAX_LIVE_EVENTS = 200
        events = cs.get("events", [])
        if len(events) > MAX_LIVE_EVENTS:
            view = dict(cs)
            trimmed = events[-MAX_LIVE_EVENTS:]
            view["events"] = [{
                "type": "other",
                "msg_type": "note",
                "content": f"… {len(events) - MAX_LIVE_EVENTS} earlier events omitted from live view (still recorded for the final step) …",
                "time": "",
            }] + trimmed
            return view
        return cs

    def _read_dag_content(self):
        """Read the latest DAG HTML file, caching by mtime."""
        if not self.latest_dag_filename:
            return None
        if (
            self._dag_content_cache is not None
            and self._dag_content_mtime == self.latest_dag_mtime
        ):
            return self._dag_content_cache
        path = os.path.join(self.working_directory, self.latest_dag_filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self._dag_content_cache = content
            self._dag_content_mtime = self.latest_dag_mtime
            return content
        except OSError:
            return self._dag_content_cache  # return stale if file vanished

    def _build_data(self, include_dag_content=False):
        """Build the data dict. include_dag_content=True for the embedded
        snapshot inside the HTML (self-contained), False for the polling
        file (kept lean)."""
        # Refresh DAG mtime
        if self.latest_dag_filename:
            try:
                self.latest_dag_mtime = os.path.getmtime(
                    os.path.join(self.working_directory, self.latest_dag_filename)
                )
            except OSError:
                pass

        dag_info = None
        if self.latest_dag_filename:
            dag_info = {
                "filename": self.latest_dag_filename,
                "mtime": self.latest_dag_mtime,
            }
            if include_dag_content:
                dag_info["html_content"] = self._read_dag_content()

        return {
            "title": self.title,
            "session_start": self.session_start,
            "last_updated": self._now_full(),
            "tokens_total": self.tokens_total,
            "steps": self.steps,
            "current_step": self._current_step_view(),
            "canvas": self._collect_canvas(),
            "explog": self._collect_explog(),
            "dag_links": {str(k): v for k, v in self.dag_links.items()},
            "latest_dag": dag_info,
            "hide_columns": {k: sorted(v) for k, v in self.hide_columns.items()},
        }

    def _flush(self, force=False):
        """Rewrite the dashboard files, subject to a time throttle.

        Called from on_event() for EVERY streamed agent event -- at least once
        per model call and once per tool call. Both files are rewritten IN FULL
        (~80 MB js + ~99 MB html measured on the 27-05 run), so unthrottled this
        was more I/O than the checkpointer itself.

        Skipping is safe: both files are views rebuilt from in-memory state, so
        the next flush emits the latest state -- nothing accumulates and nothing
        is lost. `force=True` (used by __init__ and close()) always writes, so
        the session's final state is never missing.

        The heavy self-contained HTML is throttled far harder than the lean
        live_data.js that the page actually polls.
        """
        now = time.time()
        # Kill switch beats everything, including force: "off" must mean no
        # dashboard writes at all, not "no writes except the forced ones".
        if not getattr(var, "LIVE_VIZ_ENABLED", True):
            return
        write_data = force or (now - self._last_data_flush) >= var.LIVE_VIZ_DATA_MIN_INTERVAL_S
        write_html = force or (now - self._last_html_flush) >= var.LIVE_VIZ_HTML_MIN_INTERVAL_S
        if not write_data and not write_html:
            self._skipped_flushes += 1
            return

        if write_data:
            self._last_data_flush = now
        if write_html:
            self._last_html_flush = now
        self._skipped_flushes = 0

        self._write_files(write_data=write_data, write_html=write_html)

    def _write_files(self, write_data=True, write_html=True):
        """The actual writes. Each file is independently skippable -- building
        its payload (a full json.dumps of the whole state) is most of the cost,
        so a skipped file must not be built either."""
        # 1) live_data.js (lean — no DAG content) for live polling
        if write_data:
            data_lean = self._build_data(include_dag_content=False)
            payload = "window.__LIVE_DATA__ = " + json.dumps(
                data_lean, default=str, ensure_ascii=False
            ) + ";\nwindow.__LIVE_TICK__ && window.__LIVE_TICK__();\n"

            tmp = self.data_path + ".tmp"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(payload)
                os.replace(tmp, self.data_path)
            except Exception as e:
                print(f"[LiveVisualizer] flush error (data.js): {e}")

        # 2) Rewrite the HTML with embedded data (full — includes DAG content)
        #    so downloading live_visualization.html gives a self-contained file.
        #    Much heavier than (1): it embeds every DAG, then runs a str.replace
        #    over the whole ~99 MB payload. Throttled hardest.
        if write_html:
            data_full = self._build_data(include_dag_content=True)
            data_json = json.dumps(data_full, default=str, ensure_ascii=False)
            # CRITICAL: The DAG HTML (and any other embedded content) may contain
            # </script> tags.  The browser's HTML parser sees those and thinks
            # they close OUR <script> block, breaking the page.  Escape every
            # </ sequence — this is the standard fix for inline JSON-in-script.
            data_json = data_json.replace("</", r"<\/")
            html = self._html_prefix + data_json + self._html_suffix
            tmp_html = self.html_path + ".tmp"
            try:
                with open(tmp_html, "w", encoding="utf-8") as f:
                    f.write(html)
                os.replace(tmp_html, self.html_path)
            except Exception as e:
                print(f"[LiveVisualizer] flush error (html): {e}")

    # ------------------------------------------------------------------
    # RESUME: restore state from a previous session
    # ------------------------------------------------------------------
    def _try_restore_state(self):
        """Load completed steps from an earlier run so we don't lose history."""
        if not os.path.exists(self.data_path):
            return
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                content = f.read()
            # Parse: "window.__LIVE_DATA__ = {...};\n..."
            json_str = content.split(" = ", 1)[1].split(";\n", 1)[0]
            data = json.loads(json_str)

            restored_steps = data.get("steps", [])
            if not restored_steps:
                return

            self.steps = restored_steps
            self.next_step_id = max(
                (s.get("id", 0) for s in self.steps), default=0
            ) + 1
            self.tokens_total = data.get("tokens_total", self.tokens_total)
            self.session_start = data.get("session_start", self.session_start)

            dag_links = data.get("dag_links") or {}
            self.dag_links = {}
            for k, v in dag_links.items():
                try:
                    self.dag_links[int(k)] = v
                except (ValueError, TypeError):
                    self.dag_links[k] = v

            latest = data.get("latest_dag")
            if latest and isinstance(latest, dict):
                self.latest_dag_filename = latest.get("filename")
                self.latest_dag_mtime = latest.get("mtime")

            # start a fresh streaming step for the new run
            self.current_step = self._new_step()

            print(
                f"[LiveVisualizer] Restored {len(self.steps)} steps from "
                f"previous session (next step id: {self.next_step_id})"
            )
        except Exception as e:
            print(f"[LiveVisualizer] Could not restore state: {e}")


# ==========================================================================
# HTML TEMPLATE
# ==========================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0f1117;
    --bg2: #161b27;
    --bg3: #1e2535;
    --bg4: #252d3e;
    --border: #2e3a52;
    --accent: #3b82f6;
    --accent2: #6366f1;
    --green: #22c55e;
    --yellow: #eab308;
    --red: #ef4444;
    --orange: #f97316;
    --purple: #a855f7;
    --cyan: #06b6d4;
    --text: #e2e8f0;
    --text2: #94a3b8;
    --text3: #64748b;
    --supervisor: #6366f1;
    --agent: #3b82f6;
    --tool: #22c55e;
    --boss: #a855f7;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; overflow: hidden; display: flex; flex-direction: column; }

  /* header */
  #header { background: var(--bg2); border-bottom: 1px solid var(--border); padding: 10px 18px; display: flex; align-items: center; gap: 14px; flex-shrink: 0; }
  #header h1 { font-size: 1.05rem; font-weight: 700; color: var(--text); letter-spacing: 0.01em; }
  .header-badge { background: var(--accent); color: #fff; font-size: 0.65rem; font-weight: 700; padding: 2px 8px; border-radius: 20px; letter-spacing: 0.05em; text-transform: uppercase; }
  .header-stat { color: var(--text2); font-size: 0.75rem; margin-left: auto; }
  .header-stat span { color: var(--text); font-weight: 600; }
  .live-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); display: inline-block; margin-right: 6px; box-shadow: 0 0 8px var(--green); animation: pulse 1.2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

  /* layout */
  #main { display: flex; flex: 1; overflow: hidden; position: relative; }
  #left-panel { width: 400px; min-width: 220px; max-width: 70%; display: flex; flex-direction: column; border-right: 1px solid var(--border); background: var(--bg2); overflow: hidden; }
  #right-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #canvas-panel { height: 50%; min-height: 80px; background: var(--bg2); border-bottom: 1px solid var(--border); display: flex; flex-direction: column; overflow: hidden; }
  #explog-panel { flex: 1; background: var(--bg2); display: flex; flex-direction: column; overflow: hidden; }

  .vdivider { width: 6px; cursor: col-resize; background: var(--border); flex-shrink: 0; transition: background 0.15s; position: relative; z-index: 10; }
  .vdivider:hover, .vdivider.dragging { background: var(--accent); }
  .hdivider { height: 6px; cursor: row-resize; background: var(--border); flex-shrink: 0; transition: background 0.15s; position: relative; z-index: 10; }
  .hdivider:hover, .hdivider.dragging { background: var(--accent); }

  .panel-header { display: flex; align-items: center; gap: 8px; padding: 8px 14px; background: var(--bg3); border-bottom: 1px solid var(--border); flex-shrink: 0; }
  .panel-title { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text2); }
  .panel-badge { font-size: 0.6rem; font-weight: 700; padding: 1px 7px; border-radius: 10px; background: var(--bg4); color: var(--text2); }
  .panel-badge.blue { background: #1d3a6e; color: #60a5fa; }
  .panel-badge.green { background: #14532d; color: #4ade80; }
  .panel-badge.purple { background: #3b0764; color: #c084fc; }
  .panel-badge.orange { background: #431407; color: #fb923c; }

  #step-list { flex: 1; overflow-y: auto; padding: 8px 0; }
  #step-list::-webkit-scrollbar { width: 6px; }
  #step-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  .step-card { margin: 4px 8px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg3); transition: all 0.15s; overflow: hidden; }
  .step-card:hover { border-color: var(--accent); }
  .step-card.supervisor .step-indicator { background: var(--supervisor); }
  .step-card.agent .step-indicator { background: var(--agent); }
  .step-card.boss .step-indicator { background: var(--boss); }
  .step-card.streaming { border-color: var(--green); box-shadow: 0 0 0 1px var(--green), 0 0 16px rgba(34,197,94,0.15); }
  .step-card.streaming .step-indicator { background: var(--green); animation: pulse 1.2s infinite; }
  .step-card.failed { border-color: var(--red); }
  .step-card.failed .step-indicator { background: var(--red); }

  .step-header { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; cursor: pointer; }
  .step-indicator { width: 3px; min-height: 40px; border-radius: 2px; flex-shrink: 0; align-self: stretch; }
  .step-meta { flex: 1; min-width: 0; }
  .step-num { font-size: 0.6rem; color: var(--text3); font-weight: 600; margin-bottom: 1px; display: flex; align-items: center; gap: 6px; }
  .step-title { font-size: 0.82rem; font-weight: 600; color: var(--text); line-height: 1.3; margin-bottom: 4px; }
  .step-summary { font-size: 0.71rem; color: var(--text2); line-height: 1.5; white-space: pre-wrap; }
  .step-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; align-items: center; }
  .tag { font-size: 0.58rem; font-weight: 700; padding: 1px 6px; border-radius: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
  .tag.supervisor { background: #1e1b4b; color: #a5b4fc; }
  .tag.agent { background: #1e3a5f; color: #60a5fa; }
  .tag.planning { background: #1e3a5f; color: #7dd3fc; }
  .tag.execution { background: #14532d; color: #4ade80; }
  .tag.results { background: #431407; color: #fb923c; }
  .tag.waiting { background: #3f3f46; color: #a1a1aa; }
  .tag.error { background: #450a0a; color: #fca5a5; }
  .tag.streaming-badge { background: #052e16; color: #4ade80; }

  .step-chevron { color: var(--text3); font-size: 0.7rem; margin-top: 2px; flex-shrink: 0; transition: transform 0.2s; }
  .step-card.expanded .step-chevron { transform: rotate(90deg); }
  .step-detail { display: none; padding: 0 12px 12px 25px; }
  .step-card.expanded .step-detail { display: block; }

  .detail-section { margin-top: 8px; }
  .detail-label { font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text3); margin-bottom: 4px; }
  .detail-content { font-size: 0.71rem; color: var(--text2); line-height: 1.55; background: var(--bg); border-radius: 5px; padding: 8px 10px; border: 1px solid var(--border); white-space: pre-wrap; word-break: break-word; max-height: 280px; overflow-y: auto; }
  .detail-content::-webkit-scrollbar { width: 4px; }
  .detail-content::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* tool call event */
  .ev { margin-top: 6px; border-radius: 5px; padding: 7px 10px; font-size: 0.69rem; background: #0a1628; border: 1px solid #1e3a5f; }
  .ev.tool_call { background: #0a1628; border-color: #1e3a5f; }
  .ev.tool_result { background: #0a1820; border-color: #1e4a3f; }
  .ev.ai_text { background: #1a1028; border-color: #3a1e5f; }
  .ev.structured_call { background: #1a2028; border-color: #3a4a5f; }
  .ev.streaming-ev { border-style: dashed; }
  .ev-name { font-family: monospace; font-weight: 700; color: var(--green); }
  .ev-args { color: var(--text2); font-family: monospace; font-size: 0.66rem; white-space: pre-wrap; word-break: break-word; margin-top: 3px; max-height: 120px; overflow-y: auto; }
  .ev-result { color: var(--yellow); font-family: monospace; font-size: 0.66rem; white-space: pre-wrap; word-break: break-word; margin-top: 4px; padding-top: 4px; border-top: 1px dashed #1e3a5f; max-height: 150px; overflow-y: auto; }
  .ev-pending { color: var(--text3); font-style: italic; font-size: 0.66rem; margin-top: 3px; }
  .ev-time { float: right; color: var(--text3); font-size: 0.58rem; font-family: monospace; }

  .time-chip { display: inline-flex; align-items: center; gap: 4px; font-size: 0.62rem; color: var(--cyan); background: #0a2535; border: 1px solid #0e4a6e; border-radius: 5px; padding: 1px 7px; }
  .elapsed-chip { font-size: 0.62rem; color: var(--text3); font-family: monospace; margin-left: auto; }

  /* canvas panel */
  .top-tabs { display: flex; gap: 4px; }
  .top-tab { font-size: 0.72rem; font-weight: 700; padding: 4px 12px; border-radius: 6px; cursor: pointer; color: var(--text3); background: var(--bg4); border: 1px solid var(--border); transition: all 0.15s; letter-spacing: 0.02em; }
  .top-tab:hover { color: var(--text2); }
  .top-tab.active { background: var(--bg); color: var(--accent); border-color: var(--accent); }
  .view-pane { display: none; flex-direction: column; flex: 1; overflow: hidden; }
  .view-pane.active { display: flex; }
  .dag-iframe { flex: 1; border: 0; background: var(--bg); width: 100%; height: 100%; }
  #dag-view { background: var(--bg); }

  .canvas-tabs { display: flex; gap: 2px; padding: 6px 10px 0; background: var(--bg3); overflow-x: auto; flex-shrink: 0; max-height: 100px; }
  .canvas-tabs::-webkit-scrollbar { height: 3px; }
  .canvas-tabs::-webkit-scrollbar-thumb { background: var(--border); }
  .ctab { font-size: 0.65rem; font-weight: 600; padding: 4px 10px; border-radius: 5px 5px 0 0; cursor: pointer; color: var(--text3); background: var(--bg4); border: 1px solid var(--border); border-bottom: none; white-space: nowrap; transition: all 0.15s; }
  .ctab:hover { color: var(--text2); }
  .ctab.active { background: var(--bg); color: var(--accent); border-color: var(--accent); }
  .canvas-body { flex: 1; overflow-y: auto; padding: 12px 14px; font-size: 0.73rem; line-height: 1.6; color: var(--text2); }
  .canvas-body::-webkit-scrollbar { width: 6px; }
  .canvas-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  .canvas-empty { color: var(--text3); font-style: italic; text-align: center; margin-top: 30px; }
  .canvas-body pre { font-family: 'SF Mono', 'Consolas', monospace; font-size: 0.7rem; white-space: pre-wrap; word-break: break-word; color: var(--text2); }
  .canvas-body code { background: var(--bg4); padding: 0 4px; border-radius: 3px; font-family: monospace; font-size: 0.68rem; color: var(--green); }
  .canvas-body h2 { color: var(--text); font-size: 0.85rem; margin: 12px 0 5px; border-bottom: 1px solid var(--border); padding-bottom: 3px; }
  .canvas-body h3 { color: var(--accent); font-size: 0.78rem; margin: 8px 0 3px; }
  .canvas-body strong { color: var(--text); }

  /* explog panel */
  .explog-tabs { display: flex; gap: 2px; padding: 6px 10px 0; background: var(--bg3); flex-shrink: 0; }
  .etab { font-size: 0.65rem; font-weight: 600; padding: 4px 14px; border-radius: 5px 5px 0 0; cursor: pointer; color: var(--text3); background: var(--bg4); border: 1px solid var(--border); border-bottom: none; transition: all 0.15s; }
  .etab:hover { color: var(--text2); }
  .etab.active { background: var(--bg); color: var(--purple); border-color: var(--purple); }
  .explog-body { flex: 1; overflow: auto; }
  .explog-body::-webkit-scrollbar { width: 6px; height: 6px; }
  .explog-body::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.68rem; }
  table th { background: var(--bg3); color: var(--text3); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; padding: 7px 10px; text-align: left; position: sticky; top: 0; border-bottom: 1px solid var(--border); white-space: nowrap; z-index: 2; }
  table td { padding: 6px 10px; border-bottom: 1px solid var(--border); color: var(--text2); vertical-align: top; font-size: 0.67rem; }
  table tr:hover td { background: var(--bg3); }
  table td.mono { font-family: monospace; color: var(--cyan); font-size: 0.65rem; }
  table td.num { font-family: monospace; text-align: right; color: var(--green); }
  table td.num.bad { color: var(--red); }
  table td.wrap { min-width: 200px; max-width: 340px; word-break: break-word; white-space: pre-wrap; }

  .status-badge { display: inline-block; font-size: 0.6rem; font-weight: 700; padding: 1px 7px; border-radius: 8px; text-transform: uppercase; white-space: nowrap; }
  .status-completed { background: #14532d; color: #4ade80; }
  .status-running { background: #1c3a6b; color: #60a5fa; }
  .status-pending { background: #3f3f46; color: #a1a1aa; }
  .status-failed { background: #450a0a; color: #fca5a5; font-size: 0.55rem; }

  .jtype { display: inline-block; font-size: 0.6rem; font-weight: 700; padding: 1px 6px; border-radius: 4px; }
  .jtype-bulk { background: #1e3a5f; color: #7dd3fc; }
  .jtype-surface { background: #1e1b4b; color: #a5b4fc; }
  .jtype-O { background: #431407; color: #fb923c; }
  .jtype-OH { background: #14532d; color: #4ade80; }

  .empty-table { text-align: center; color: var(--text3); font-style: italic; padding: 30px; font-size: 0.8rem; }

  #search-wrap { padding: 8px; border-bottom: 1px solid var(--border); flex-shrink: 0; }
  #search { width: 100%; background: var(--bg3); border: 1px solid var(--border); color: var(--text); padding: 6px 10px; border-radius: 6px; font-size: 0.75rem; outline: none; }
  #search:focus { border-color: var(--accent); }
  #search::placeholder { color: var(--text3); }

  /* markdown */
  .md-h2 { color: var(--text); font-size: 0.8rem; font-weight: 700; margin: 6px 0 3px; }
  .md-h3 { color: var(--accent); font-size: 0.74rem; font-weight: 600; margin: 4px 0 2px; }
  .md-bold { color: var(--text); font-weight: 700; }

  .dag-link { font-size: 0.63rem; color: var(--cyan); background: #0a2535; border: 1px solid #0e4a6e; border-radius: 4px; padding: 1px 6px; text-decoration: none; }
  .dag-link:hover { background: #0e4a6e; }
</style>
</head>
<body>

<div id="header">
  <div class="header-badge">LIVE</div>
  <h1 id="title">__TITLE__</h1>
  <span class="time-chip" id="session-time">📅 —</span>
  <div class="header-stat">
    <span class="live-dot"></span>
    Updated <span id="last-updated">—</span>
    &nbsp;|&nbsp; Steps: <span id="total-steps">0</span>
    &nbsp;|&nbsp; Tokens: <span id="total-tokens">0 / 0</span>
  </div>
</div>

<div id="main">
  <!-- LEFT PANEL: STEP LOG -->
  <div id="left-panel">
    <div class="panel-header">
      <span class="panel-title">🗂 Session Log</span>
      <span class="panel-badge blue" id="step-count-badge">0 steps</span>
      <span class="panel-badge green" id="streaming-badge" style="display:none">● streaming</span>
    </div>
    <div id="search-wrap">
      <input type="text" id="search" placeholder="🔍  Search steps..." oninput="filterSteps(this.value)">
    </div>
    <div id="step-list"></div>
  </div>

  <div class="vdivider" id="vdiv"></div>

  <!-- RIGHT PANEL -->
  <div id="right-panel">
    <!-- CANVAS / DAG PANEL -->
    <div id="canvas-panel">
      <div class="panel-header">
        <div class="top-tabs">
          <div class="top-tab active" id="top-tab-canvas" onclick="switchTopTab('canvas')">🖼 Canvas</div>
          <div class="top-tab" id="top-tab-dag" onclick="switchTopTab('dag')">🕸 DAG</div>
        </div>
        <span class="panel-badge purple" id="canvas-key-count">0 keys</span>
        <span class="panel-badge" id="dag-file-label" style="display:none"></span>
        <span style="margin-left:auto; font-size:0.65rem; color:var(--text3)">live</span>
      </div>
      <div id="canvas-view" class="view-pane active">
        <div class="canvas-tabs" id="canvas-tabs"></div>
        <div class="canvas-body" id="canvas-body"><div class="canvas-empty">Canvas is empty.</div></div>
      </div>
      <div id="dag-view" class="view-pane">
        <div id="dag-empty" class="canvas-empty">No DAG has been generated yet.</div>
        <iframe id="dag-iframe" class="dag-iframe" style="display:none"></iframe>
      </div>
    </div>

    <div class="hdivider" id="hdiv"></div>

    <!-- EXPLOG PANEL -->
    <div id="explog-panel">
      <div class="panel-header">
        <span class="panel-title">📊 EXPLOG</span>
        <span class="panel-badge green" id="explog-cand-count">0 candidates</span>
        <span class="panel-badge orange" id="explog-proc-count" style="margin-left:4px">0 processes</span>
        <span style="margin-left:auto; font-size:0.65rem; color:var(--text3)">live</span>
      </div>
      <div class="explog-tabs">
        <div class="etab active" id="etab-cand" onclick="switchExplogTab('candidates')">Candidates</div>
        <div class="etab" id="etab-proc" onclick="switchExplogTab('processes')">Processes</div>
      </div>
      <div class="explog-body" id="explog-body"><div class="empty-table">EXPLOG is empty.</div></div>
    </div>
  </div>
</div>

<script>
// ============================================================
// EMBEDDED DATA (replaced by Python on every flush — self-contained)
// ============================================================
window.__EMBEDDED_DATA__ = /* __EMBEDDED_DATA_JSON__ */null;

// ============================================================
// STATE
// ============================================================
let DATA = null;
let expandedSteps = new Set();      // ids of user-expanded completed steps
let userCollapsedStreaming = new Set(); // ids of streaming steps user manually collapsed
let currentExplogTab = 'candidates';
let currentCanvasTab = null;
let currentTopTab = 'canvas';       // 'canvas' or 'dag'
let lastDagKey = '';                // filename|mtime seen on iframe, to detect changes
let lastSignature = '';
let searchFilter = '';
let stickyBottom = true;   // auto-scroll step list only if already at bottom
let isLive = false;                 // true once polling succeeds at least once
let embeddedDagContent = null;      // DAG HTML from embedded snapshot (for offline)

// Grab DAG content from embedded data before polling can overwrite it
if (window.__EMBEDDED_DATA__ && window.__EMBEDDED_DATA__.latest_dag) {
  embeddedDagContent = window.__EMBEDDED_DATA__.latest_dag.html_content || null;
}

// ============================================================
// INITIAL RENDER from embedded data (works offline / downloaded)
// ============================================================
if (window.__EMBEDDED_DATA__) {
  DATA = window.__EMBEDDED_DATA__;
  setTimeout(render, 0);
}

// ============================================================
// POLLING (via script-tag reload so it works with file:// URLs)
// ============================================================
const POLL_MS = __POLL_MS__;
const DATA_FILE = '__DATA_FILE__';

function pollOnce() {
  const s = document.createElement('script');
  s.src = DATA_FILE + '?t=' + Date.now();
  s.onload = () => { s.remove(); };
  s.onerror = () => { s.remove(); };
  document.head.appendChild(s);
}
window.__LIVE_TICK__ = function() {
  isLive = true;
  DATA = window.__LIVE_DATA__;
  render();
};
setInterval(pollOnce, POLL_MS);
pollOnce();

// ============================================================
// RENDER
// ============================================================
function render() {
  if (!DATA) return;
  const sig = JSON.stringify({
    steps: DATA.steps.length,
    current: DATA.current_step ? (DATA.current_step.events || []).length + ':' + (DATA.current_step.agent || '') : null,
    canvas: Object.keys(DATA.canvas || {}),
    cand: (DATA.explog && DATA.explog.candidates ? DATA.explog.candidates.length : 0),
    proc: (DATA.explog && DATA.explog.processes ? DATA.explog.processes.length : 0),
    tokens: DATA.tokens_total,
    upd: DATA.last_updated,
    dag: DATA.latest_dag ? (DATA.latest_dag.filename + '|' + DATA.latest_dag.mtime) : null,
  });
  if (sig === lastSignature) return;
  lastSignature = sig;

  // header
  document.getElementById('title').textContent = DATA.title || 'Agent Run';
  document.getElementById('session-time').textContent = '📅 ' + (DATA.session_start || '—');
  document.getElementById('last-updated').textContent = DATA.last_updated || '—';
  const allSteps = [...(DATA.steps || [])];
  if (DATA.current_step) allSteps.push(DATA.current_step);
  document.getElementById('total-steps').textContent = allSteps.length;
  const tks = DATA.tokens_total || {input:0,output:0};
  document.getElementById('total-tokens').textContent = (tks.input||0).toLocaleString() + ' in / ' + (tks.output||0).toLocaleString() + ' out';
  document.getElementById('step-count-badge').textContent = (DATA.steps.length) + ' complete';
  document.getElementById('streaming-badge').style.display = DATA.current_step ? '' : 'none';

  renderSteps();
  renderCanvas();
  renderDag();
  renderExplog();
}

// ============================================================
// STEPS
// ============================================================
function renderSteps() {
  const listEl = document.getElementById('step-list');
  // remember scroll position
  const atBottom = (listEl.scrollHeight - listEl.scrollTop - listEl.clientHeight) < 50;

  const allSteps = [...(DATA.steps || [])];
  if (DATA.current_step) allSteps.push(DATA.current_step);

  const filtered = allSteps.filter(st => {
    if (!searchFilter) return true;
    const q = searchFilter.toLowerCase();
    return (st.title||'').toLowerCase().includes(q)
      || (st.summary||'').toLowerCase().includes(q)
      || (st.agent||'').toLowerCase().includes(q)
      || (st.phase||'').toLowerCase().includes(q);
  });

  listEl.innerHTML = filtered.map(renderStepCard).join('');

  // re-bind toggle handlers
  listEl.querySelectorAll('.step-card').forEach(card => {
    const id = Number(card.dataset.id);
    card.querySelector('.step-header').addEventListener('click', () => toggleStep(id));
  });

  if (stickyBottom && atBottom) {
    listEl.scrollTop = listEl.scrollHeight;
  }
}

function renderStepCard(st) {
  const isStreaming = (st.status === 'streaming');
  const isFailed = (st.tags || []).includes('error') && !(st.tags || []).includes('waiting');
  const agent = st.agent || '…';
  const kind = agent.toLowerCase().includes('supervisor') ? 'supervisor'
             : agent.toLowerCase().includes('boss') ? 'boss'
             : 'agent';
  // expansion rule: streaming is expanded unless user collapsed; completed is collapsed unless user expanded
  let expanded;
  if (isStreaming) expanded = !userCollapsedStreaming.has(st.id);
  else expanded = expandedSteps.has(st.id);

  const classes = ['step-card', kind];
  if (isStreaming) classes.push('streaming');
  if (isFailed) classes.push('failed');
  if (expanded) classes.push('expanded');

  const tagsHtml = (st.tags || []).map(t => `<span class="tag ${escapeHtml(t)}">${escapeHtml(t)}</span>`).join('');
  const statusTag = isStreaming ? `<span class="tag streaming-badge">● live</span>` : '';

  const agentColor = kind === 'supervisor' ? '#a5b4fc' : kind === 'boss' ? '#c084fc' : '#60a5fa';
  const phase = st.phase || (isStreaming ? 'streaming…' : '');
  const elapsed = st.end_time && st.start_time ? elapsedBetween(st.start_time, st.end_time) : '';
  const dagLink = (DATA.dag_links && DATA.dag_links[st.id])
    ? `<a class="dag-link" href="${DATA.dag_links[st.id]}" target="_blank" onclick="event.stopPropagation()">DAG ↗</a>` : '';

  const title = escapeHtml(st.title || (isStreaming ? `${agent} — working…` : agent));
  const summary = st.summary ? escapeHtml(st.summary) : (isStreaming ? `<span style="color:var(--text3);font-style:italic">…streaming tool calls…</span>` : '');

  return `
    <div class="${classes.join(' ')}" data-id="${st.id}">
      <div class="step-header">
        <div class="step-indicator"></div>
        <div class="step-meta">
          <div class="step-num">
            STEP ${st.id} · <span style="color:${agentColor}">${escapeHtml(agent)}</span>
            ${phase ? '· ' + escapeHtml(phase) : ''}
            ${dagLink}
            ${elapsed ? `<span class="elapsed-chip">${elapsed}</span>` : ''}
          </div>
          <div class="step-title">${title}</div>
          <div class="step-summary">${summary}</div>
          <div class="step-tags">${tagsHtml} ${statusTag}
            <span class="time-chip" style="margin-left:auto">${escapeHtml(st.start_time || '')}</span>
          </div>
        </div>
        <div class="step-chevron">▶</div>
      </div>
      <div class="step-detail">
        ${renderStepEvents(st)}
        ${renderStructuredResponse(st)}
        ${st.detail ? `<div class="detail-section"><div class="detail-label">Notes</div><div class="detail-content">${renderMarkdown(st.detail)}</div></div>` : ''}
      </div>
    </div>
  `;
}

function renderStepEvents(st) {
  const events = st.events || [];
  if (!events.length) return '';
  const rows = events.map(ev => {
    const t = ev.time ? `<span class="ev-time">${escapeHtml(ev.time)}</span>` : '';
    if (ev.type === 'tool_call') {
      const pending = ev.result === null;
      const cls = 'ev tool_call' + (pending && st.status === 'streaming' ? ' streaming-ev' : '');
      const resultHtml = pending
        ? `<div class="ev-pending">⏳ waiting for result…</div>`
        : `<div class="ev-result">${escapeHtml(ev.result_full || ev.result || '')}</div>`;
      return `<div class="${cls}">
        ${t}<span class="ev-name">🔧 ${escapeHtml(ev.name || '')}</span>
        ${ev.args_preview ? `<div class="ev-args">${escapeHtml(ev.args_full || ev.args_preview)}</div>` : ''}
        ${resultHtml}
      </div>`;
    }
    if (ev.type === 'tool_result') {
      return `<div class="ev tool_result">
        ${t}<span class="ev-name">⬅ ${escapeHtml(ev.name || '')}</span>
        <div class="ev-result">${escapeHtml(ev.content_full || ev.content || '')}</div>
      </div>`;
    }
    if (ev.type === 'ai_text') {
      return `<div class="ev ai_text">${t}<span class="ev-name">💬 assistant</span>
        <div class="ev-args">${escapeHtml(ev.content || '')}</div></div>`;
    }
    if (ev.type === 'structured_call') {
      return `<div class="ev structured_call">${t}<span class="ev-name">📦 ${escapeHtml(ev.name || '')}</span>
        <div class="ev-args">${escapeHtml(ev.args_preview || '')}</div></div>`;
    }
    if (ev.type === 'human_input') {
      return `<div class="ev">${t}<span class="ev-name">👤 human</span>
        <div class="ev-args">${escapeHtml(ev.content || '')}</div></div>`;
    }
    return `<div class="ev">${t}<span class="ev-name">${escapeHtml(ev.type || '?')}</span>
      <div class="ev-args">${escapeHtml(ev.content || '')}</div></div>`;
  }).join('');
  return `<div class="detail-section">
    <div class="detail-label">Events (${events.length})</div>
    <div class="detail-content" style="background:transparent;border:none;padding:0;max-height:none">${rows}</div>
  </div>`;
}

function renderStructuredResponse(st) {
  const sr = st.structured_response;
  if (!sr) return '';
  const p = sr.parsed || {};
  let inner = '';
  if (sr.type === 'Act') {
    if (p.action_type === 'Plan') {
      inner = '<ol style="padding-left:18px;margin:4px 0">' +
        (p.steps || []).map(s => {
          const step = typeof s === 'string' ? s : (s.step || '');
          const ag = typeof s === 'object' ? (s.agent || '') : '';
          return `<li style="margin:3px 0"><span style="color:var(--text)">${escapeHtml(step)}</span>${ag ? ` <span style="color:var(--text3);font-size:0.65rem">[${escapeHtml(ag)}]</span>` : ''}</li>`;
        }).join('') + '</ol>';
    } else if (p.action_type === 'NoChange') {
      inner = `<div>${escapeHtml(p.comment || '')}</div>`;
    } else if (p.action_type === 'Response') {
      inner = `<div style="color:var(--text)">${escapeHtml(p.response || '')}</div>`;
    } else if (p.action_type === 'Error') {
      inner = `<div style="color:var(--red)">${escapeHtml(p.error || '')}</div>`;
    } else {
      inner = `<pre>${escapeHtml(JSON.stringify(p, null, 2))}</pre>`;
    }
  } else if (sr.type === 'wokerResponse') {
    const success = p.success !== false;
    inner = `<div><span style="color:${success ? 'var(--green)' : 'var(--red)'};font-weight:700">${success ? '✅ success' : '❌ failed'}</span></div>
      <div style="margin-top:6px"><b>answer:</b> ${escapeHtml(p.answer || '')}</div>
      ${p.summary ? `<div style="margin-top:6px"><b>summary:</b> ${escapeHtml(p.summary)}</div>` : ''}`;
  } else if (sr.type === 'BossReview') {
    const approve = (p.decision || '').toLowerCase() === 'approve';
    inner = `<div><span style="color:${approve ? 'var(--green)' : 'var(--yellow)'};font-weight:700">${approve ? '✅ APPROVED' : '↩ REVISE'}</span></div>
      ${p.feedback ? `<div style="margin-top:6px">${escapeHtml(p.feedback)}</div>` : ''}`;
  } else {
    inner = `<pre>${escapeHtml(sr.raw || '')}</pre>`;
  }
  return `<div class="detail-section">
    <div class="detail-label">Structured response — ${escapeHtml(sr.type || '')}</div>
    <div class="detail-content">${inner}</div>
  </div>`;
}

function toggleStep(id) {
  // Find step to know if it's streaming
  const all = [...(DATA.steps || [])];
  if (DATA.current_step) all.push(DATA.current_step);
  const st = all.find(s => s.id === id);
  if (!st) return;
  if (st.status === 'streaming') {
    if (userCollapsedStreaming.has(id)) userCollapsedStreaming.delete(id);
    else userCollapsedStreaming.add(id);
  } else {
    if (expandedSteps.has(id)) expandedSteps.delete(id);
    else expandedSteps.add(id);
  }
  renderSteps();
}

function filterSteps(val) {
  searchFilter = val;
  renderSteps();
}

// ============================================================
// CANVAS
// ============================================================
function renderCanvas() {
  const canvas = DATA.canvas || {};
  const keys = Object.keys(canvas);
  document.getElementById('canvas-key-count').textContent = keys.length + ' key' + (keys.length !== 1 ? 's' : '');
  const tabsEl = document.getElementById('canvas-tabs');
  const bodyEl = document.getElementById('canvas-body');
  if (!keys.length) {
    tabsEl.innerHTML = '';
    bodyEl.innerHTML = '<div class="canvas-empty">🔲 Canvas is empty.</div>';
    return;
  }
  if (!currentCanvasTab || !keys.includes(currentCanvasTab)) currentCanvasTab = keys[keys.length - 1];
  tabsEl.innerHTML = keys.map(k =>
    `<div class="ctab ${k === currentCanvasTab ? 'active' : ''}" onclick="switchCanvasTab('${escapeAttr(k)}')">${escapeHtml(k)}</div>`
  ).join('');
  renderCanvasContent(currentCanvasTab);
}

function switchCanvasTab(k) {
  currentCanvasTab = k;
  document.querySelectorAll('.ctab').forEach(t => t.classList.toggle('active', t.textContent === k));
  renderCanvasContent(k);
}

function switchTopTab(view) {
  currentTopTab = view;
  document.getElementById('top-tab-canvas').classList.toggle('active', view === 'canvas');
  document.getElementById('top-tab-dag').classList.toggle('active', view === 'dag');
  document.getElementById('canvas-view').classList.toggle('active', view === 'canvas');
  document.getElementById('dag-view').classList.toggle('active', view === 'dag');
  // badges only relevant to the active view
  document.getElementById('canvas-key-count').style.display = view === 'canvas' ? '' : 'none';
  document.getElementById('dag-file-label').style.display = view === 'dag' ? '' : 'none';
  if (view === 'dag') renderDag(true);
}

function renderDag(force) {
  const latest = DATA && DATA.latest_dag;
  const iframe = document.getElementById('dag-iframe');
  const empty = document.getElementById('dag-empty');
  const label = document.getElementById('dag-file-label');

  if (!latest || !latest.filename) {
    iframe.style.display = 'none';
    empty.style.display = '';
    empty.textContent = 'No DAG has been generated yet.';
    label.textContent = '';
    return;
  }

  label.textContent = latest.filename;
  const key = latest.filename + '|' + (latest.mtime || '');

  if (key !== lastDagKey || force) {
    lastDagKey = key;
    iframe.style.display = '';
    empty.style.display = 'none';

    if (isLive) {
      // Live mode: load the DAG file directly (it sits next to the HTML).
      // Cache-bust via the file's mtime so the browser always re-fetches.
      iframe.removeAttribute('srcdoc');
      iframe.src = latest.filename + '?t=' + encodeURIComponent(latest.mtime || Date.now());
    } else {
      // Offline / downloaded mode: use the embedded DAG HTML content.
      const content = (latest.html_content) || embeddedDagContent;
      if (content) {
        iframe.removeAttribute('src');
        iframe.srcdoc = content;
      } else {
        iframe.removeAttribute('srcdoc');
        iframe.src = latest.filename + '?t=' + Date.now();  // best-effort
      }
    }
  }
}

function renderCanvasContent(key) {
  const bodyEl = document.getElementById('canvas-body');
  const canvas = (DATA && DATA.canvas) || {};
  const raw = canvas[key];
  if (raw === undefined) {
    bodyEl.innerHTML = `<div class="canvas-empty">No data for <code>${escapeHtml(key)}</code></div>`;
    return;
  }
  bodyEl.innerHTML = renderMarkdown(raw);
}

// ============================================================
// EXPLOG
// ============================================================
function switchExplogTab(tab) {
  currentExplogTab = tab;
  document.getElementById('etab-cand').classList.toggle('active', tab === 'candidates');
  document.getElementById('etab-proc').classList.toggle('active', tab === 'processes');
  renderExplog();
}

function renderExplog() {
  const ex = DATA.explog || {candidates: [], processes: []};
  document.getElementById('explog-cand-count').textContent = (ex.candidates||[]).length + ' candidate' + ((ex.candidates||[]).length !== 1 ? 's' : '');
  document.getElementById('explog-proc-count').textContent = (ex.processes||[]).length + ' process' + ((ex.processes||[]).length !== 1 ? 'es' : '');
  const rows = currentExplogTab === 'candidates' ? ex.candidates : ex.processes;
  const body = document.getElementById('explog-body');
  if (!rows || !rows.length) {
    body.innerHTML = `<div class="empty-table">No ${currentExplogTab} in EXPLOG yet.</div>`;
    return;
  }
  // auto-derive columns, but drop any that appear in hide_columns for this tab
  const hide = new Set(((DATA.hide_columns || {})[currentExplogTab]) || []);
  const cols = Object.keys(rows[0]).filter(c => !hide.has(c));
  const head = '<tr>' + cols.map(c => `<th>${escapeHtml(c)}</th>`).join('') + '</tr>';
  const tbody = rows.map(r => {
    return '<tr>' + cols.map(c => {
      const v = r[c];
      if (v === null || v === undefined || v === '') return '<td><span style="color:var(--text3)">—</span></td>';
      // Job type gets a color-coded badge instead of a plain cell
      if (c === 'job_type') {
        return `<td>${renderJobTypeBadge(v)}</td>`;
      }
      // Status gets a color-coded badge too
      if (c === 'status') {
        return `<td>${renderStatusBadge(v)}</td>`;
      }
      const cls = cellClassFor(c, v);
      return `<td class="${cls}">${escapeHtml(renderCellValue(c, v))}</td>`;
    }).join('') + '</tr>';
  }).join('');
  body.innerHTML = `<table><thead>${head}</thead><tbody>${tbody}</tbody></table>`;
}

function renderJobTypeBadge(v) {
  const s = String(v);
  let cls = 'jtype';
  if (s === 'bulk_relaxation')         cls += ' jtype-bulk';
  else if (s === 'surface_relaxation') cls += ' jtype-surface';
  else if (s === 'O_adsorption')       cls += ' jtype-O';
  else if (s === 'OH_adsorption')      cls += ' jtype-OH';
  return `<span class="${cls}">${escapeHtml(s.replace('_', ' '))}</span>`;
}

function renderStatusBadge(v) {
  const s = String(v).toLowerCase();
  let cls = 'status-badge ';
  if (s === 'completed')       cls += 'status-completed';
  else if (s === 'running')    cls += 'status-running';
  else if (s === 'pending')    cls += 'status-pending';
  else if (s.includes('fail') || s.includes('unrecoverable') || s.includes('error')) cls += 'status-failed';
  else cls += 'status-pending';
  const label = (s.includes('fail') || s.includes('unrecoverable') || s.includes('error')) ? 'FAILED' : String(v);
  return `<span class="${cls}" title="${escapeHtml(String(v))}">${escapeHtml(label)}</span>`;
}

function renderCellValue(col, v) {
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) return String(v);
    return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/\.?0+$/, '');
  }
  return String(v);
}

function cellClassFor(col, v) {
  const c = col.toLowerCase();
  if (c.includes('id') && typeof v === 'string' && v.length >= 8) return 'mono';
  if (c === 'processnote') return 'wrap wide-note';
  if (c === 'status') return '';
  if (typeof v === 'number') {
    if (c.includes('overpot') || c.includes('deviation')) {
      if (Math.abs(v) > 1.0) return 'num bad';
      return 'num';
    }
    return 'num';
  }
  if (c.includes('note') || c.includes('reason')) return 'wrap';
  return '';
}

// ============================================================
// HELPERS
// ============================================================
function escapeHtml(s) {
  if (s === null || s === undefined) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
function escapeAttr(s) { return escapeHtml(s); }

function renderMarkdown(text) {
  if (text === null || text === undefined) return '';
  let s = String(text);
  // escape first
  s = escapeHtml(s);
  s = s
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+?)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<div class="md-h3">$1</div>')
    .replace(/^## (.+)$/gm, '<div class="md-h2">$1</div>')
    .replace(/→/g, '<span style="color:var(--accent)">→</span>')
    .replace(/✅/g, '<span style="color:var(--green)">✅</span>')
    .replace(/❌/g, '<span style="color:var(--red)">❌</span>')
    .replace(/⚠️/g, '<span style="color:var(--yellow)">⚠️</span>')
    .replace(/🔄/g, '<span style="color:var(--accent)">🔄</span>')
    .replace(/⭐/g, '<span style="color:var(--yellow)">⭐</span>')
    .replace(/\n/g, '<br>');
  return s;
}

function elapsedBetween(startStr, endStr) {
  try {
    const s = new Date(startStr.replace(' ', 'T'));
    const e = new Date(endStr.replace(' ', 'T'));
    const ms = e - s;
    if (isNaN(ms) || ms < 0) return '';
    const sec = Math.floor(ms / 1000);
    if (sec < 60) return sec + 's';
    const m = Math.floor(sec / 60), r = sec % 60;
    if (m < 60) return m + 'm ' + r + 's';
    const h = Math.floor(m / 60), mm = m % 60;
    return h + 'h ' + mm + 'm';
  } catch (e) { return ''; }
}

// ============================================================
// RESIZERS
// ============================================================
function initResizers() {
  const vdiv = document.getElementById('vdiv');
  const leftPanel = document.getElementById('left-panel');
  const main = document.getElementById('main');
  let vDragging = false, vStartX = 0, vStartW = 0;
  vdiv.addEventListener('mousedown', e => {
    vDragging = true; vStartX = e.clientX; vStartW = leftPanel.offsetWidth;
    vdiv.classList.add('dragging'); document.body.style.userSelect = 'none'; document.body.style.cursor = 'col-resize';
  });
  document.addEventListener('mousemove', e => {
    if (!vDragging) return;
    const dx = e.clientX - vStartX;
    const newW = Math.max(220, Math.min(vStartW + dx, main.offsetWidth * 0.75));
    leftPanel.style.width = newW + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (vDragging) { vDragging = false; vdiv.classList.remove('dragging'); document.body.style.userSelect=''; document.body.style.cursor=''; }
  });

  const hdiv = document.getElementById('hdiv');
  const canvasPanel = document.getElementById('canvas-panel');
  const rightPanel = document.getElementById('right-panel');
  let hDragging = false, hStartY = 0, hStartH = 0;
  hdiv.addEventListener('mousedown', e => {
    hDragging = true; hStartY = e.clientY; hStartH = canvasPanel.offsetHeight;
    hdiv.classList.add('dragging'); document.body.style.userSelect = 'none'; document.body.style.cursor = 'row-resize';
  });
  document.addEventListener('mousemove', e => {
    if (!hDragging) return;
    const dy = e.clientY - hStartY;
    const newH = Math.max(80, Math.min(hStartH + dy, rightPanel.offsetHeight - 80));
    canvasPanel.style.height = newH + 'px'; canvasPanel.style.flex = 'none';
  });
  document.addEventListener('mouseup', () => {
    if (hDragging) { hDragging = false; hdiv.classList.remove('dragging'); document.body.style.userSelect=''; document.body.style.cursor=''; }
  });
}
initResizers();

// Save user scroll intent so auto-scroll only runs when user is at bottom
document.getElementById('step-list').addEventListener('scroll', () => {
  const el = document.getElementById('step-list');
  stickyBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < 50;
});
</script>
</body>
</html>
"""
