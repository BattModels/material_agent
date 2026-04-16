"""
DAG Visualizer
==============
Builds a Directed Acyclic Graph (DAG) from a list of unordered artifact nodes
and generates a self-contained HTML file to visualize it.

Each node is expected to have:
  - .result_id            : unique string identifier
  - .parent_result_ids    : list of parent node result_ids ([] for root nodes)
  - .tool_name            : display name shown on the node
  - .value                : node value (shown in popup)
  - .reasons              : Dict[param_name, reason] (shown in popup)
  - .description          : node description (shown in side panel)
  - .metadata             : arbitrary dict (shown in side panel)

Usage:
    from dag_visualizer import build_dag, generate_html, save_html

    nodes = [...]                        # your list of artifact nodes
    dag   = build_dag(nodes)             # build adjacency structure
    html  = generate_html(dag)           # render to HTML string
    save_html(html, "dag.html")          # write to disk
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import json
import html as _html


# ---------------------------------------------------------------------------
# 1.  DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class DAGNode:
    """Lightweight wrapper around a raw artifact node."""
    id: str                              # == raw.result_id
    parent_ids: list[str]
    raw: Any = field(default=None, repr=False)

    # populated by build_dag()
    children_ids: list[str] = field(default_factory=list, repr=False)
    depth: int = 0


@dataclass
class DAG:
    nodes: dict[str, DAGNode]           # result_id -> DAGNode
    roots: list[str]                    # result_ids with no parents
    topological_order: list[str]        # BFS topological order


# ---------------------------------------------------------------------------
# 2.  CYCLE ERROR
# ---------------------------------------------------------------------------

class CycleError(ValueError):
    """
    Raised when the node graph contains one or more cycles.

    Attributes
    ----------
    cycle_node_ids : set[str]
        IDs of every node that is part of (or only reachable from) a cycle.
    cycles : list[list[str]]
        Each inner list is one minimal cycle path, e.g. ["A","B","C","A"].
    """

    def __init__(self, cycle_node_ids: set[str], nodes: dict[str, "DAGNode"]):
        self.cycle_node_ids = cycle_node_ids
        self.cycles         = self._find_cycles(cycle_node_ids, nodes)
        super().__init__(self._build_message(nodes))

    # ------------------------------------------------------------------
    # cycle finder: DFS on the subgraph of cycle-involved nodes
    # ------------------------------------------------------------------
    @staticmethod
    def _find_cycles(
        candidate_ids: set[str],
        nodes: dict[str, "DAGNode"],
    ) -> list[list[str]]:
        """
        Return a list of simple cycles found by DFS within *candidate_ids*.
        Each cycle is represented as a list starting and ending with the
        same node id, e.g. ["A", "B", "C", "A"].
        """
        found:   list[list[str]] = []
        visited: set[str]        = set()

        def dfs(nid: str, path: list[str], path_set: set[str]) -> None:
            visited.add(nid)
            path.append(nid)
            path_set.add(nid)

            for child_id in nodes[nid].children_ids:
                if child_id not in candidate_ids:
                    continue
                if child_id in path_set:
                    # found a back-edge → extract the cycle
                    cycle_start = path.index(child_id)
                    cycle = path[cycle_start:] + [child_id]
                    # de-duplicate: normalise by rotating to smallest id
                    body   = cycle[:-1]
                    min_i  = body.index(min(body))
                    normed = body[min_i:] + body[:min_i] + [body[min_i]]
                    if normed not in found:
                        found.append(normed)
                elif child_id not in visited:
                    dfs(child_id, path, path_set)

            path.pop()
            path_set.discard(nid)

        for start in candidate_ids:
            if start not in visited:
                dfs(start, [], set())

        return found

    # ------------------------------------------------------------------
    # human-readable message
    # ------------------------------------------------------------------
    def _build_message(self, nodes: dict[str, "DAGNode"]) -> str:
        lines: list[str] = []
        lines.append(
            f"Cycle detected — the graph is NOT a DAG.\n"
            f"{len(self.cycle_node_ids)} node(s) are involved in or blocked by cycles."
        )

        # ── per-cycle detail ───────────────────────────────────────────
        lines.append(f"\n{'─'*60}")
        lines.append(f"  Cycles found: {len(self.cycles)}")
        lines.append(f"{'─'*60}")
        for i, cycle in enumerate(self.cycles, 1):
            arrow_path = " → ".join(cycle)
            lines.append(f"\n  Cycle {i}  ({len(cycle)-1} node(s))")
            lines.append(f"    Path : {arrow_path}")

            # show each edge in the cycle with tool_name if available
            lines.append("    Edges:")
            for a, b in zip(cycle, cycle[1:]):
                ta = getattr(nodes[a].raw, "tool_name", a) if nodes.get(a) else a
                tb = getattr(nodes[b].raw, "tool_name", b) if nodes.get(b) else b
                lines.append(f"      {a} ({ta})  →  {b} ({tb})")

        # ── full list of trapped nodes ─────────────────────────────────
        lines.append(f"\n{'─'*60}")
        lines.append(f"  All {len(self.cycle_node_ids)} trapped node(s):")
        lines.append(f"{'─'*60}")
        for nid in sorted(self.cycle_node_ids):
            raw       = nodes[nid].raw
            tool_name = getattr(raw, "tool_name", "?")
            parents   = nodes[nid].parent_ids
            children  = nodes[nid].children_ids
            in_cycle  = any(nid in c for c in self.cycles)
            flag      = "● IN CYCLE" if in_cycle else "○ blocked"
            lines.append(
                f"  {flag}  {nid}  [{tool_name}]\n"
                f"           parents : {parents}\n"
                f"           children: {children}"
            )

        # ── fix hints ─────────────────────────────────────────────────
        lines.append(f"\n{'─'*60}")
        lines.append("  How to fix:")
        lines.append(f"{'─'*60}")
        for i, cycle in enumerate(self.cycles, 1):
            body = cycle[:-1]
            lines.append(f"  Cycle {i}: remove one of these edges to break it:")
            for a, b in zip(cycle, cycle[1:]):
                lines.append(f"    • {b}.parent_result_ids.remove('{a}')")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3.  BUILD THE DAG
# ---------------------------------------------------------------------------

def build_dag(raw_nodes: list[Any]) -> DAG:
    """
    Build a DAG from an unordered list of artifact nodes.

    Expects each node to expose:
        node.result_id           -> str
        node.parent_result_ids   -> list[str]

    Isolated nodes (no parents and no children after the graph is fully
    wired) are silently removed before the DAG is returned.  A
    ``UserWarning`` is issued listing the removed IDs so callers can
    audit the data if needed.
    """
    nodes: dict[str, DAGNode] = {}
    for raw in raw_nodes:
        nid = raw.result_id
        parent_ids = list(raw.parent_result_ids or [])
        nodes[nid] = DAGNode(id=nid, parent_ids=parent_ids, raw=raw)

    # build child pointers
    for nid, node in nodes.items():
        for pid in node.parent_ids:
            if pid in nodes:
                nodes[pid].children_ids.append(nid)

    # roots = nodes with no known parents
    roots = [nid for nid, n in nodes.items() if not n.parent_ids]

    # Kahn's topological BFS + depth assignment
    in_degree: dict[str, int] = {nid: len(n.parent_ids) for nid, n in nodes.items()}
    queue: deque[str] = deque(r for r in nodes if in_degree[r] == 0)
    topo: list[str] = []

    while queue:
        nid = queue.popleft()
        topo.append(nid)
        for child_id in nodes[nid].children_ids:
            nodes[child_id].depth = max(nodes[child_id].depth, nodes[nid].depth + 1)
            in_degree[child_id] -= 1
            if in_degree[child_id] == 0:
                queue.append(child_id)

    if len(topo) != len(nodes):
        # Nodes still in the graph after Kahn's algorithm are exactly those
        # involved in (or reachable only from) one or more cycles.
        cycle_node_ids = {nid for nid in nodes if nid not in set(topo)}
        raise CycleError(cycle_node_ids, nodes)

    # ------------------------------------------------------------------
    # Remove isolated nodes: no parents AND no children.
    # This step runs after the full graph is wired so that a node whose
    # declared parent doesn't exist in the input is still treated as a
    # potential root, not silently dropped here.
    # ------------------------------------------------------------------
    isolated: set[str] = {
        nid for nid, n in nodes.items()
        if not n.parent_ids and not n.children_ids
    }
    if isolated:
        import warnings
        warnings.warn(
            f"[build_dag] Removed {len(isolated)} isolated node(s) "
            f"(no parents, no children): {', '.join(sorted(isolated))}",
            UserWarning,
            stacklevel=2,
        )
        for nid in isolated:
            del nodes[nid]
        roots = [r for r in roots if r not in isolated]
        topo  = [nid for nid in topo  if nid not in isolated]

    return DAG(nodes=nodes, roots=roots, topological_order=topo)


# ---------------------------------------------------------------------------
# 4.  BUILD GRAPH DATA FOR vis-network
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    """Return a JSON-serialisable version of v."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, dict):
        return {str(k): _safe(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_safe(i) for i in v]
    return str(v)


def _build_graph_data(dag: DAG) -> tuple[list[dict], list[dict], dict, list[str]]:
    """
    Convert DAG into vis-network nodes/edges dicts plus a rich nodeData map
    (keyed by result_id) that carries all extra fields for the popup / panel,
    and a chronological order list (nodes sorted by creation timestamp).

    Vis-network level assignment
    ----------------------------
    Roots are ordered by their ``timestamp`` attribute (earliest → lowest level).
    This means a root created later sinks below earlier roots in the top-down
    hierarchy.  Non-root node levels are propagated in topological order so that
    every node appears at least one level below its deepest parent.
    """
    # ── 1. Compute vis_level for every node ───────────────────────────────
    def _ts(nid: str) -> float:
        return float(getattr(dag.nodes[nid].raw, "timestamp", 0) or 0)

    # Roots: sorted ascending by timestamp → rank 0 = earliest (top of graph)
    sorted_roots = sorted(dag.roots, key=_ts)
    vis_level: dict[str, int] = {nid: rank for rank, nid in enumerate(sorted_roots)}

    # Non-roots: propagate in topological order (guarantees all parents done first)
    for nid in dag.topological_order:
        if nid in vis_level:
            continue
        parent_levels = [vis_level.get(pid, 0) for pid in dag.nodes[nid].parent_ids]
        vis_level[nid] = (max(parent_levels) if parent_levels else 0) + 1

    # ── 2. Build chronological order (ascending timestamp) ────────────────
    chrono_order: list[str] = sorted(dag.topological_order, key=_ts)

    # ── 3. Build vis-network node / edge dicts and nodeData map ──────────
    vis_nodes: list[dict] = []
    node_data: dict[str, dict] = {}

    for nid, node in dag.nodes.items():
        raw = node.raw

        tool_name    = getattr(raw, "tool_name",    nid)
        value        = getattr(raw, "value",         None)
        listed_value = getattr(raw, "listed_value",  False)
        reasons      = getattr(raw, "reasons",       {})
        description  = getattr(raw, "description",   "")
        metadata     = getattr(raw, "metadata",      {})
        timestamp    = getattr(raw, "timestamp",     None)

        # Expand listed artifacts: value is an iterable of objects with .value
        if listed_value and value is not None:
            try:
                value = [item.value for item in value]
            except (TypeError, AttributeError):
                pass   # fall back to the raw value unchanged

        vis_nodes.append({
            "id":    nid,
            "label": str(tool_name),
            "level": vis_level.get(nid, node.depth),
            "title": None,   # disable vis built-in tooltip
        })

        node_data[nid] = {
            "result_id":    nid,
            "tool_name":    str(tool_name),
            "depth":        node.depth,
            "timestamp":    float(timestamp) if timestamp is not None else None,
            "parent_ids":   node.parent_ids,
            "children_ids": node.children_ids,
            "value":        _safe(value),
            "listed_value": bool(listed_value),
            "reasons":      _safe(reasons) if isinstance(reasons, dict) else {},
            "description":  str(description) if description else "",
            "metadata":     _safe(metadata)  if isinstance(metadata, dict) else {},
        }

    vis_edges: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for nid, node in dag.nodes.items():
        for pid in node.parent_ids:
            key = (pid, nid)
            if key not in seen:
                seen.add(key)
                vis_edges.append({"from": pid, "to": nid, "arrows": "to"})

    return vis_nodes, vis_edges, node_data, chrono_order


# ---------------------------------------------------------------------------
# 5.  GENERATE HTML
# ---------------------------------------------------------------------------

def generate_html(dag: DAG, title: str = "DAG Visualizer") -> str:
    """Return a self-contained HTML string visualizing the DAG."""
    vis_nodes, vis_edges, node_data, chrono_order = _build_graph_data(dag)

    nodes_json       = json.dumps(vis_nodes,    indent=2)
    edges_json       = json.dumps(vis_edges,    indent=2)
    node_data_json   = json.dumps(node_data,    indent=2)
    chrono_json      = json.dumps(chrono_order)
    roots_json       = json.dumps(dag.roots)

    num_nodes = len(dag.nodes)
    num_edges = len(vis_edges)
    num_roots = len(dag.roots)
    max_depth = max((n.depth for n in dag.nodes.values()), default=0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{_html.escape(title)}</title>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
  <link  href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" />

  <style>
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}

    :root {{
      --bg:         #0d0f14;
      --surface:    #151821;
      --border:     #252a38;
      --accent:     #4f8ef7;
      --accent-dim: #1e3a6e;
      --accent2:    #7f5af0;
      --text:       #e2e8f0;
      --muted:      #64748b;
      --green:      #4ade80;
      --green-dim:  #14301a;
      --pin-border: #f59e0b;
      --pin-dim:    #2d1f00;
      --font-mono:  'Fira Code','Cascadia Code','Consolas',monospace;
      --font-sans:  'DM Sans','Segoe UI',sans-serif;
    }}

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Fira+Code:wght@400;500&display=swap');

    body {{
      background:var(--bg); color:var(--text); font-family:var(--font-sans);
      height:100vh; display:flex; flex-direction:column; overflow:hidden;
    }}

    /* ── Header ── */
    header {{
      display:flex; align-items:center; justify-content:space-between;
      padding:14px 24px; background:var(--surface);
      border-bottom:1px solid var(--border); gap:24px; flex-shrink:0;
    }}
    .brand {{ display:flex; align-items:center; gap:10px; }}
    .brand-icon {{
      width:32px; height:32px;
      background:linear-gradient(135deg,var(--accent),var(--accent2));
      border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:16px;
    }}
    h1 {{ font-size:15px; font-weight:600; letter-spacing:.02em; }}
    .stats {{ display:flex; gap:6px; }}
    .stat {{
      background:var(--bg); border:1px solid var(--border); border-radius:6px;
      padding:4px 12px; font-size:12px; font-family:var(--font-mono); color:var(--muted);
    }}
    .stat span {{ color:var(--accent); font-weight:500; }}
    .toolbar {{ display:flex; gap:8px; }}
    .btn {{
      background:var(--surface); border:1px solid var(--border); color:var(--muted);
      border-radius:6px; padding:5px 12px; font-size:12px; cursor:pointer;
      transition:all .15s; font-family:var(--font-sans);
    }}
    .btn:hover {{ border-color:var(--accent); color:var(--accent); }}

    /* ── Main layout ── */
    main {{ display:flex; flex:1; overflow:hidden; position:relative; }}

    #graph {{ flex:1; background:var(--bg); position:relative; }}
    #graph::before {{
      content:''; position:absolute; inset:0;
      background-image:
        linear-gradient(var(--border) 1px,transparent 1px),
        linear-gradient(90deg,var(--border) 1px,transparent 1px);
      background-size:40px 40px; opacity:.3; pointer-events:none;
    }}

    /* ── SVG overlay for connecting strings ── */
    #svg-overlay {{
      position:fixed; top:0; left:0; width:100vw; height:100vh;
      pointer-events:none; z-index:998;
    }}

    /* ── Side panel ── */
    aside {{
      width:300px; background:var(--surface); border-left:1px solid var(--border);
      display:flex; flex-direction:column; overflow:hidden; flex-shrink:0;
    }}
    .panel-title {{
      font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase;
      color:var(--muted); padding:14px 16px 8px; border-bottom:1px solid var(--border);
    }}
    #node-detail {{ padding:16px; font-size:13px; flex:1; overflow-y:auto; min-height:0; }}
    .detail-empty {{ color:var(--muted); font-style:italic; font-size:12px; margin-top:4px; }}
    .detail-field {{ margin-bottom:14px; }}
    .detail-label {{
      font-size:10px; text-transform:uppercase; letter-spacing:.08em;
      color:var(--muted); margin-bottom:4px;
    }}
    .detail-value {{
      font-family:var(--font-mono); font-size:12px; color:var(--text);
      background:var(--bg); border:1px solid var(--border);
      border-radius:5px; padding:6px 8px; word-break:break-all;
    }}
    .detail-text {{
      font-size:12px; color:var(--text); line-height:1.6;
      background:var(--bg); border:1px solid var(--border);
      border-radius:5px; padding:8px; white-space:pre-wrap; word-break:break-word;
    }}
    .tag-list {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:4px; }}
    .tag {{
      background:var(--accent-dim); color:var(--accent);
      border-radius:4px; padding:2px 7px; font-family:var(--font-mono); font-size:11px;
    }}
    .tag.green {{ background:var(--green-dim); color:var(--green); }}
    .kv-table {{ width:100%; border-collapse:collapse; font-size:11px; font-family:var(--font-mono); }}
    .kv-table td {{
      padding:4px 6px; border:1px solid var(--border);
      vertical-align:top; word-break:break-word;
    }}
    .kv-table td:first-child {{ color:var(--accent); white-space:nowrap; width:35%; }}

    /* ── Topo list ── */
    #topo-list {{
      padding:8px 16px 16px; overflow-y:auto;
      display:flex; flex-direction:column; gap:3px;
      max-height:200px; min-height:0; flex-shrink:0;
    }}
    .topo-item {{
      display:flex; align-items:center; gap:8px; padding:5px 8px; border-radius:5px;
      cursor:pointer; transition:background .1s; font-size:12px; font-family:var(--font-mono);
    }}
    .topo-item:hover {{ background:var(--border); }}
    .topo-item.selected {{ background:var(--accent-dim); color:var(--accent); }}
    .topo-idx {{ font-size:10px; color:var(--muted); min-width:22px; text-align:right; }}
    .topo-id  {{ flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
    .depth-badge {{
      font-size:10px; background:var(--border); border-radius:3px; padding:1px 5px; color:var(--muted);
    }}

    /* ── Shared popup anatomy ── */
    .popup-base {{
      position:fixed; z-index:1000; width:340px;
      background:var(--surface);
      border-radius:10px; box-shadow:0 8px 32px #00000090;
      overflow:hidden; font-size:13px;
      transition:opacity .12s, transform .12s;
    }}
    .popup-header {{
      display:flex; align-items:center; justify-content:space-between;
      padding:9px 13px; cursor:move; user-select:none;
      border-bottom:1px solid var(--border); gap:8px;
    }}
    .popup-tool-name {{
      font-family:var(--font-mono); font-size:13px; font-weight:600;
      flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
    }}
    .popup-hint {{ font-size:10px; color:var(--muted); letter-spacing:.04em; flex-shrink:0; }}
    .popup-close-btn {{
      background:none; border:none; color:var(--muted); cursor:pointer;
      font-size:15px; line-height:1; padding:0 2px; transition:color .15s; flex-shrink:0;
    }}
    .popup-close-btn:hover {{ color:var(--text); }}
    .popup-body {{
      padding:12px 14px; display:flex; flex-direction:column; gap:10px;
      max-height:380px; overflow-y:auto;
    }}
    .popup-field {{ display:flex; flex-direction:column; gap:3px; }}
    .popup-label {{ font-size:10px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); }}
    .popup-value {{
      font-family:var(--font-mono); font-size:12px; color:var(--text);
      background:var(--bg); border:1px solid var(--border);
      border-radius:4px; padding:5px 8px; word-break:break-all; white-space:pre-wrap;
    }}
    .popup-divider {{ height:1px; background:var(--border); margin:2px 0; }}

    /* hover popup: blue border, semi-transparent */
    #hover-popup {{
      border:1px solid var(--accent);
      pointer-events:none;         /* never intercepts mouse during hover */
      opacity:0; transform:translateY(4px);
    }}
    #hover-popup.visible {{
      opacity:1; transform:translateY(0);
    }}
    #hover-popup .popup-header {{ background:var(--accent-dim); cursor:default; }}
    #hover-popup .popup-tool-name {{ color:var(--accent); }}

    /* pinned popup: amber border, fully interactive */
    .pinned-popup {{
      border:1px solid var(--pin-border);
    }}
    .pinned-popup .popup-header {{ background:var(--pin-dim); }}
    .pinned-popup .popup-tool-name {{ color:var(--pin-border); }}
    .pin-icon {{ font-size:13px; flex-shrink:0; }}

    /* scrollbar */
    ::-webkit-scrollbar {{ width:5px; }}
    ::-webkit-scrollbar-track {{ background:transparent; }}
    ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
  </style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-icon">⬡</div>
    <h1>{_html.escape(title)}</h1>
  </div>
  <div class="stats">
    <div class="stat">nodes <span>{num_nodes}</span></div>
    <div class="stat">edges <span>{num_edges}</span></div>
    <div class="stat">roots <span>{num_roots}</span></div>
    <div class="stat">depth <span>{max_depth}</span></div>
  </div>
  <div class="toolbar">
    <button class="btn" onclick="network.fit()">⊡ Fit</button>
    <button class="btn" onclick="toggleLayout()">⇅ Layout</button>
    <button class="btn" onclick="settlePhysics()">⟳ Settle</button>
  </div>
</header>

<main>
  <div id="graph"></div>

  <aside>
    <div class="panel-title">Node Detail</div>
    <div id="node-detail">
      <div class="detail-empty">Click a node to inspect it.</div>
    </div>
    <div class="panel-title" style="border-top:1px solid var(--border)">Chronological Order</div>
    <div id="topo-list"></div>
  </aside>
</main>

<!-- SVG overlay: connecting strings from pinned popups to nodes -->
<svg id="svg-overlay" xmlns="http://www.w3.org/2000/svg"></svg>

<!-- Hover popup — shows on hover, hides on blur, never pinned -->
<div id="hover-popup" class="popup-base">
  <div class="popup-header">
    <span class="popup-tool-name" id="hover-tool-name"></span>
    <span class="popup-hint">click to pin</span>
  </div>
  <div class="popup-body" id="hover-body"></div>
</div>

<script>
// ── Injected data ─────────────────────────────────────────────────────────
const RAW_NODES   = {nodes_json};
const RAW_EDGES   = {edges_json};
const NODE_DATA   = {node_data_json};
const CHRONO_ORDER = {chrono_json};
const ROOT_IDS    = new Set({roots_json});

// ── vis-network ───────────────────────────────────────────────────────────
const container = document.getElementById("graph");

const visNodes = RAW_NODES.map(n => ({{
  ...n,
  shape:"box",
  borderWidth: ROOT_IDS.has(n.id) ? 2 : 1.5,
  color:{{
    background: ROOT_IDS.has(n.id) ? "#1a2a1a":"#1e2535",
    border:     ROOT_IDS.has(n.id) ? "#4ade80":"#4f8ef7",
    highlight:{{ background: ROOT_IDS.has(n.id)?"#22381e":"#253050",
                 border:     ROOT_IDS.has(n.id)?"#86efac":"#93c5fd" }},
    hover:    {{ background: ROOT_IDS.has(n.id)?"#1e2e1c":"#202c47",
                 border:     ROOT_IDS.has(n.id)?"#6ee77a":"#7eb5fc" }},
  }},
  font:{{
    color:"#e2e8f0", face:"'Fira Code',monospace", size:13,
    bold: ROOT_IDS.has(n.id) ? {{color:"#4ade80",mod:"bold"}} : false,
  }},
  margin:{{top:8,bottom:8,left:12,right:12}},
  shadow:{{enabled:true,color:"#00000060",size:10,x:2,y:2}},
  title: undefined,
}}));

const visEdges = RAW_EDGES.map(e => ({{
  ...e,
  color:{{color:"#4f8ef780",highlight:"#4f8ef7",hover:"#7eb5fc"}},
  smooth:{{type:"cubicBezier",forceDirection:"vertical",roundness:0.5}},
  width:1.5, selectionWidth:2.5,
}}));

let isHierarchical = true;

const options = {{
  layout:{{
    hierarchical:{{
      enabled:true, direction:"UD", sortMethod:"directed",
      levelSeparation:100, nodeSpacing:140, treeSpacing:200,
    }},
  }},
  physics:{{enabled:false}},
  interaction:{{
    hover:true,
    tooltipDelay:99999,   // effectively disable vis tooltip
    navigationButtons:false,
    keyboard:true,
  }},
  edges:{{arrows:{{to:{{enabled:true,scaleFactor:0.7}}}}}},
}};

const dsNodes = new vis.DataSet(visNodes);
const dsEdges = new vis.DataSet(visEdges);
const network = new vis.Network(container, {{nodes:dsNodes, edges:dsEdges}}, options);

function settlePhysics() {{
  network.setOptions({{physics:{{enabled:true}}}});
  setTimeout(() => network.setOptions({{physics:{{enabled:false}}}}), 1500);
}}
function toggleLayout() {{
  isHierarchical = !isHierarchical;
  network.setOptions({{
    layout:{{hierarchical:{{
      enabled:isHierarchical, direction:"UD", sortMethod:"directed",
      levelSeparation:100, nodeSpacing:140, treeSpacing:200,
    }}}},
    physics:{{enabled:!isHierarchical}},
  }});
  if (!isHierarchical) setTimeout(()=>network.setOptions({{physics:{{enabled:false}}}}),1500);
  network.fit();
}}

// ── Helpers ───────────────────────────────────────────────────────────────
function escHtml(s) {{
  return String(s ?? "")
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}}
function renderKV(obj, emptyMsg) {{
  if (!obj || typeof obj !== "object" || !Object.keys(obj).length)
    return `<span style="color:var(--muted);font-size:12px">${{emptyMsg??"—"}}</span>`;
  return `<table class="kv-table"><tbody>` +
    Object.entries(obj)
      .map(([k,v])=>`<tr><td>${{escHtml(k)}}</td><td>${{escHtml(
        typeof v==="object"?JSON.stringify(v,null,2):v
      )}}</td></tr>`).join("") +
    `</tbody></table>`;
}}
function fmtTimestamp(ts) {{
  if (ts === null || ts === undefined) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleString(undefined, {{
    year:"numeric", month:"short", day:"2-digit",
    hour:"2-digit", minute:"2-digit", second:"2-digit", fractionalSecondDigits:3,
  }});
}}
function renderValue(d) {{
  if (d.value === null || d.value === undefined) return `<div class="popup-value">—</div>`;
  if (d.listed_value && Array.isArray(d.value)) {{
    // Render each item as a numbered row
    const rows = d.value.map((v, i) =>
      `<tr><td style="color:var(--muted);text-align:right;padding-right:8px;white-space:nowrap">${{i+1}}</td>` +
      `<td>${{escHtml(typeof v==="object"?JSON.stringify(v,null,2):String(v))}}</td></tr>`
    ).join("");
    return `<table class="kv-table" style="margin-top:2px"><tbody>${{rows}}</tbody></table>`;
  }}
  const valStr = typeof d.value==="object"
    ? JSON.stringify(d.value, null, 2) : String(d.value);
  return `<div class="popup-value">${{escHtml(valStr)}}</div>`;
}}
function popupBodyHTML(d) {{
  return `
    <div class="popup-field">
      <div class="popup-label">Result ID</div>
      <div class="popup-value">${{escHtml(d.result_id)}}</div>
    </div>
    <div class="popup-divider"></div>
    <div class="popup-field">
      <div class="popup-label">Created</div>
      <div class="popup-value">${{fmtTimestamp(d.timestamp)}}</div>
    </div>
    <div class="popup-divider"></div>
    <div class="popup-field">
      <div class="popup-label">Value${{d.listed_value?" (list)":""}}</div>
      ${{renderValue(d)}}
    </div>
    <div class="popup-divider"></div>
    <div class="popup-field">
      <div class="popup-label">Reasons</div>
      ${{renderKV(d.reasons,"No reasons provided")}}
    </div>`;
}}

// ── Node viewport position ────────────────────────────────────────────────
function nodeViewportPos(nid) {{
  const cr  = container.getBoundingClientRect();
  const dom = network.canvasToDOM(network.getPosition(nid));
  return {{ x: cr.left + dom.x, y: cr.top + dom.y }};
}}

// ── Hover popup ───────────────────────────────────────────────────────────
// Uses mousemove + getNodeAt() — far more reliable than hoverNode/blurNode events
const hoverPopup    = document.getElementById("hover-popup");
const hoverToolName = document.getElementById("hover-tool-name");
const hoverBody     = document.getElementById("hover-body");
let   hoverNid      = null;

function placeHoverPopup(clientX, clientY) {{
  const pw = 340, ph = hoverPopup.offsetHeight || 240;
  const vw = window.innerWidth, vh = window.innerHeight;
  let left = clientX + 52, top = clientY - 80;
  if (left + pw > vw - 8) left = clientX - pw - 52;
  if (top + ph > vh - 8)  top  = vh - ph - 8;
  if (top < 8) top = 8;
  hoverPopup.style.left = left + "px";
  hoverPopup.style.top  = top  + "px";
}}

container.addEventListener("mousemove", e => {{
  const nid = network.getNodeAt({{ x: e.offsetX, y: e.offsetY }});
  if (nid !== undefined && nid !== null) {{
    if (hoverNid !== nid) {{
      hoverNid = nid;
      const d = NODE_DATA[nid];
      hoverToolName.textContent = d.tool_name;
      hoverBody.innerHTML = popupBodyHTML(d);
    }}
    placeHoverPopup(e.clientX, e.clientY);
    hoverPopup.classList.add("visible");
  }} else {{
    hoverPopup.classList.remove("visible");
    hoverNid = null;
  }}
}});

container.addEventListener("mouseleave", () => {{
  hoverPopup.classList.remove("visible");
  hoverNid = null;
}});

// ── Pinned popups ─────────────────────────────────────────────────────────
// PINNED[nid] = {{ el: HTMLElement, path: SVGPathElement }}
const PINNED     = {{}};
const svgOverlay = document.getElementById("svg-overlay");
let   zTop       = 1000;

function nextZ() {{ return ++zTop; }}

/** Nearest point on popup border toward (tx, ty). */
function borderPoint(rect, tx, ty) {{
  const cx = rect.left + rect.width  / 2;
  const cy = rect.top  + rect.height / 2;
  const dx = tx - cx, dy = ty - cy;
  if (dx === 0 && dy === 0) return {{ x: cx, y: cy }};
  let t = Infinity;
  if (dx > 0)  t = Math.min(t, (rect.right  - cx) / dx);
  else if (dx < 0) t = Math.min(t, (rect.left - cx) / dx);
  if (dy > 0)  t = Math.min(t, (rect.bottom - cy) / dy);
  else if (dy < 0) t = Math.min(t, (rect.top  - cy) / dy);
  return {{ x: cx + dx * t, y: cy + dy * t }};
}}

function buildPath(rect, nx, ny) {{
  const bp  = borderPoint(rect, nx, ny);
  // control point: midpoint pulled slightly perpendicular for a gentle curve
  const mx  = (bp.x + nx) / 2;
  const my  = (bp.y + ny) / 2;
  const len = Math.hypot(nx - bp.x, ny - bp.y);
  const perp = len * 0.2;
  // perpendicular offset (rotate direction 90°)
  const ux = -(ny - bp.y) / (len || 1);
  const uy =  (nx - bp.x) / (len || 1);
  const cpx = mx + ux * perp;
  const cpy = my + uy * perp;
  return `M ${{bp.x.toFixed(1)}},${{bp.y.toFixed(1)}} Q ${{cpx.toFixed(1)}},${{cpy.toFixed(1)}} ${{nx.toFixed(1)}},${{ny.toFixed(1)}}`;
}}

function updateLine(nid) {{
  const entry = PINNED[nid];
  if (!entry) return;
  const pr = entry.el.getBoundingClientRect();
  if (pr.width === 0) return;
  const {{ x: nx, y: ny }} = nodeViewportPos(nid);
  entry.path.setAttribute("d", buildPath(pr, nx, ny));
}}

// Continuous rAF loop — keeps all lines correct during drag, pan, zoom, animation
function lineLoop() {{
  Object.keys(PINNED).forEach(updateLine);
  requestAnimationFrame(lineLoop);
}}
requestAnimationFrame(lineLoop);

function makeDraggable(el, nid) {{
  let drag = false, ox = 0, oy = 0;
  const header = el.querySelector(".popup-header");
  header.addEventListener("mousedown", e => {{
    if (e.target.classList.contains("popup-close-btn")) return;
    drag = true;
    el.style.zIndex = nextZ();
    const r = el.getBoundingClientRect();
    ox = e.clientX - r.left;
    oy = e.clientY - r.top;
    e.preventDefault();
  }});
  document.addEventListener("mousemove", e => {{
    if (!drag) return;
    el.style.left   = (e.clientX - ox) + "px";
    el.style.top    = (e.clientY - oy) + "px";
    el.style.right  = "auto";
    el.style.bottom = "auto";
  }});
  document.addEventListener("mouseup", () => {{ drag = false; }});
}}

function pinNode(nid, spawnX, spawnY) {{
  if (PINNED[nid]) {{
    PINNED[nid].el.style.zIndex = nextZ();
    return;
  }}

  const d = NODE_DATA[nid];

  const el = document.createElement("div");
  el.className    = "popup-base pinned-popup";
  el.style.zIndex = nextZ();
  el.innerHTML = `
    <div class="popup-header">
      <span class="pin-icon">📌</span>
      <span class="popup-tool-name">${{escHtml(d.tool_name)}}</span>
      <span class="popup-hint">drag to move</span>
      <button class="popup-close-btn" title="Close">✕</button>
    </div>
    <div class="popup-body">${{popupBodyHTML(d)}}</div>
  `;
  document.body.appendChild(el);

  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("stroke", "#f59e0b");
  path.setAttribute("stroke-width", "1.8");
  path.setAttribute("stroke-dasharray", "6 3");
  path.setAttribute("fill", "none");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("opacity", "0.85");
  svgOverlay.appendChild(path);

  PINNED[nid] = {{ el, path }};

  // spawn offset from click point
  const pw = 340;
  const vw = window.innerWidth, vh = window.innerHeight;
  let left = spawnX + 24, top = spawnY - 50;
  if (left + pw > vw - 8) left = spawnX - pw - 24;
  if (top < 8) top = 8;
  if (top + 360 > vh - 8) top = vh - 368;
  el.style.left = left + "px";
  el.style.top  = top  + "px";

  el.querySelector(".popup-close-btn").addEventListener("click", () => {{
    el.remove();
    path.remove();
    delete PINNED[nid];
  }});

  makeDraggable(el, nid);
}}

// ── Network events ────────────────────────────────────────────────────────
network.on("click", params => {{
  if (!params.nodes.length) return;
  const nid    = params.nodes[0];
  const domPos = params.event.center;
  showDetail(nid);
  pinNode(nid, domPos.x, domPos.y);
}});

// ── Side panel ────────────────────────────────────────────────────────────
let selectedId = null;

function showDetail(nid) {{
  const d = NODE_DATA[nid];
  if (!d) return;
  selectedId = nid;

  const isRoot    = ROOT_IDS.has(nid);
  const parentTags = d.parent_ids.length
    ? d.parent_ids.map(p=>`<span class="tag">${{escHtml(p)}}</span>`).join("")
    : `<span style="color:var(--muted);font-size:12px">none</span>`;
  const childTags  = d.children_ids.length
    ? d.children_ids.map(c=>`<span class="tag">${{escHtml(c)}}</span>`).join("")
    : `<span style="color:var(--muted);font-size:12px">none</span>`;

  document.getElementById("node-detail").innerHTML = `
    <div class="detail-field">
      <div class="detail-label">Tool Name</div>
      <div class="detail-value">${{escHtml(d.tool_name)}}</div>
    </div>
    <div class="detail-field">
      <div class="detail-label">Result ID</div>
      <div class="detail-value">${{escHtml(d.result_id)}}</div>
    </div>
    <div class="detail-field">
      <div class="detail-label">Depth</div>
      <div class="detail-value">${{d.depth}}</div>
    </div>
    <div class="detail-field">
      <div class="detail-label">Type</div>
      <div class="tag-list">
        ${{isRoot?'<span class="tag green">root</span>':'<span class="tag">non-root</span>'}}
      </div>
    </div>
    <div class="detail-field">
      <div class="detail-label">Parents (${{d.parent_ids.length}})</div>
      <div class="tag-list">${{parentTags}}</div>
    </div>
    <div class="detail-field">
      <div class="detail-label">Children (${{d.children_ids.length}})</div>
      <div class="tag-list">${{childTags}}</div>
    </div>
    <div class="detail-field">
      <div class="detail-label">Description</div>
      ${{d.description
        ?`<div class="detail-text">${{escHtml(d.description)}}</div>`
        :`<span style="color:var(--muted);font-size:12px">—</span>`
      }}
    </div>
    <div class="detail-field">
      <div class="detail-label">Metadata</div>
      ${{renderKV(d.metadata,"No metadata")}}
    </div>
  `;

  document.querySelectorAll(".topo-item").forEach(el=>
    el.classList.toggle("selected", el.dataset.id===nid));
  const sel = document.querySelector(`.topo-item[data-id="${{nid}}"]`);
  if (sel) sel.scrollIntoView({{block:"nearest"}});
}}

// ── Chronological order list ──────────────────────────────────────────────
const topoList = document.getElementById("topo-list");
CHRONO_ORDER.forEach((nid, i) => {{
  const d   = NODE_DATA[nid];
  const div = document.createElement("div");
  div.className  = "topo-item";
  div.dataset.id = nid;
  const tsStr = d.timestamp
    ? new Date(d.timestamp * 1000).toLocaleTimeString(undefined, {{
        hour:"2-digit", minute:"2-digit", second:"2-digit"
      }})
    : "—";
  div.innerHTML  = `
    <span class="topo-idx">${{i+1}}</span>
    <span class="topo-id">${{escHtml(d.tool_name)}}</span>
    <span class="depth-badge" title="Created ${{tsStr}}">${{tsStr}}</span>
  `;
  div.addEventListener("click", () => {{
    network.selectNodes([nid]);
    network.focus(nid, {{scale:1.2, animation:true}});
    const {{ x, y }} = nodeViewportPos(nid);
    showDetail(nid);
    pinNode(nid, x, y);
  }});
  topoList.appendChild(div);
}});

// ── Two-finger pan & pinch-to-zoom ────────────────────────────────────────
let _touchPrev  = null;   // [{{x,y}}, {{x,y}}] — previous frame's two touch points
let _pinchPrev  = null;   // previous distance between the two fingers

container.addEventListener("touchstart", e => {{
  if (e.touches.length === 2) {{
    e.preventDefault();
    _touchPrev = [
      {{ x: e.touches[0].clientX, y: e.touches[0].clientY }},
      {{ x: e.touches[1].clientX, y: e.touches[1].clientY }},
    ];
    _pinchPrev = Math.hypot(
      e.touches[1].clientX - e.touches[0].clientX,
      e.touches[1].clientY - e.touches[0].clientY,
    );
  }}
}}, {{ passive: false }});

container.addEventListener("touchmove", e => {{
  if (e.touches.length !== 2) return;
  e.preventDefault();

  const t0 = {{ x: e.touches[0].clientX, y: e.touches[0].clientY }};
  const t1 = {{ x: e.touches[1].clientX, y: e.touches[1].clientY }};

  // ── Pan: translate by midpoint delta ──────────────────────────────────
  if (_touchPrev) {{
    const prevMid = {{
      x: (_touchPrev[0].x + _touchPrev[1].x) / 2,
      y: (_touchPrev[0].y + _touchPrev[1].y) / 2,
    }};
    const curMid = {{ x: (t0.x + t1.x) / 2, y: (t0.y + t1.y) / 2 }};
    const dx = curMid.x - prevMid.x;
    const dy = curMid.y - prevMid.y;
    const scale = network.getScale();
    const pos   = network.getViewPosition();
    network.moveTo({{
      position: {{ x: pos.x - dx / scale, y: pos.y - dy / scale }},
      animation: false,
    }});
  }}

  // ── Zoom: scale by pinch ratio, anchored at midpoint ──────────────────
  const dist = Math.hypot(t1.x - t0.x, t1.y - t0.y);
  if (_pinchPrev && _pinchPrev > 0) {{
    const ratio    = dist / _pinchPrev;
    const newScale = Math.min(Math.max(network.getScale() * ratio, 0.05), 5);
    network.moveTo({{ scale: newScale, animation: false }});
  }}

  _touchPrev = [t0, t1];
  _pinchPrev = dist;
}}, {{ passive: false }});

container.addEventListener("touchend", e => {{
  if (e.touches.length < 2) {{
    _touchPrev = null;
    _pinchPrev = null;
  }}
}});

network.once("afterDrawing", () => network.fit({{animation:true}}));
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 6.  SAVE HELPER
# ---------------------------------------------------------------------------

def save_html(html: str, path: str = "dag.html") -> None:
    """Write the HTML string to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"[dag_visualizer] HTML written to: {path}")


# ---------------------------------------------------------------------------
# 7.  DEMO
# ---------------------------------------------------------------------------

class _MockNode:
    """Minimal stand-in for a real artifact node."""
    def __init__(self, result_id, parent_ids, tool_name,
                 value=None, reasons=None, description="", metadata=None,
                 timestamp=None, listed_value=False):
        self.result_id         = result_id
        self.parent_result_ids = parent_ids
        self.tool_name         = tool_name
        self.value             = value
        self.reasons           = reasons or {}
        self.description       = description
        self.metadata          = metadata or {}
        self.timestamp         = timestamp
        self.listed_value      = listed_value


class _MockListedItem:
    """Stand-in for an item inside a ListedArtifact.value."""
    def __init__(self, value):
        self.value = value


def _demo():
    import time
    t0 = time.time()

    # Simulate nodes created at different times (t0 + N seconds)
    nodes = [
        _MockNode("rid-A", [], "fetch_data",
                  value="raw_dataset_v3",
                  description="Entry point — fetches raw data from upstream source.",
                  metadata={"source": "s3://bucket/data.csv", "rows": 50000},
                  timestamp=t0 + 0.0),

        # Second root created LATER → should appear below rid-A in the layout
        _MockNode("rid-X", [], "side_input",
                  value="external_ref",
                  description="A second root created later; appears below rid-A.",
                  timestamp=t0 + 1.5),

        _MockNode("rid-B", ["rid-A"], "preprocess",
                  description="Cleans and tokenises raw data.",
                  metadata={"steps": ["strip", "lower", "tokenise"]},
                  timestamp=t0 + 2.0),
        _MockNode("rid-C", ["rid-A"], "embed",
                  value=[0.12, -0.34, 0.99],
                  reasons={"model": "embedding model selected for domain"},
                  description="Generates vector embeddings from preprocessed text.",
                  metadata={"model": "text-embed-3", "dims": 3},
                  timestamp=t0 + 2.5),
        _MockNode("rid-D", ["rid-A"], "classify",
                  value="positive",
                  reasons={"threshold": "0.5 cutoff applied"},
                  description="Runs sentiment classification on raw input.",
                  metadata={"confidence": 0.91},
                  timestamp=t0 + 3.0),
        _MockNode("rid-E", ["rid-B", "rid-C"], "merge_results",
                  value={"merged": True},
                  reasons={"input_a": "needed for join", "input_b": "provides context"},
                  description="Merges outputs from B and C into a unified result.",
                  metadata={"version": "1.2", "author": "pipeline"},
                  timestamp=t0 + 4.0),
        _MockNode("rid-F", ["rid-C", "rid-D", "rid-X"], "aggregate",
                  value=42.7,
                  reasons={"weight": "used for scoring", "baseline": "normalisation ref"},
                  description="Aggregates C, D, and side_input into a single score.",
                  timestamp=t0 + 4.5),
        _MockNode("rid-G", ["rid-E", "rid-F"], "rank",
                  # listed_value=True: value is a list of objects with .value
                  value=[_MockListedItem("result_alpha"), _MockListedItem("result_beta"),
                         _MockListedItem("result_gamma")],
                  listed_value=True,
                  description="Ranks merged results by aggregate score. Value is a ListedArtifact.",
                  timestamp=t0 + 5.0),
        _MockNode("rid-H", ["rid-G"], "format_output",
                  value="<html>...</html>",
                  description="Formats the ranked list as HTML.",
                  metadata={"template": "report_v2"},
                  timestamp=t0 + 6.0),
        _MockNode("rid-I", ["rid-G"], "store",
                  value=True,
                  description="Persists ranking to database.",
                  metadata={"db": "postgres", "table": "results"},
                  timestamp=t0 + 6.2),
        _MockNode("rid-J", ["rid-H", "rid-I"], "notify",
                  value="email_sent",
                  reasons={"trigger": "downstream completion", "channel": "email preferred"},
                  description="Sends notification once all downstream tasks complete.",
                  metadata={"recipients": ["team@example.com"]},
                  timestamp=t0 + 7.0),
        # Isolated node — no parents, no children — should be silently dropped
        _MockNode("rid-Z", [], "orphan_node",
                  description="This node has no connections and should be removed.",
                  metadata={"note": "isolated"},
                  timestamp=t0 + 0.5),
    ]

    dag = build_dag(nodes)
    print("Chronological order:", sorted(dag.topological_order,
          key=lambda n: getattr(dag.nodes[n].raw, "timestamp", 0)))
    print("Roots              :", dag.roots)
    html = generate_html(dag, title="Sample DAG")
    save_html(html, "dag.html")


if __name__ == "__main__":
    _demo()
